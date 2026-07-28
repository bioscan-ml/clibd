"""使用 CBG RIG 数据作为 query、BIOSCAN-5M 作为 reference 的检索流水线。

流水线与原有训练和评估入口相互独立：

    prepare -> embed -> index -> search -> evaluate

每个阶段都会保存可复用的中间结果，并可通过
``cbg_retrieval.stage=<stage>`` 单独运行。代码复用现有 CLIBD 的模型加载和
embedding 归一化逻辑；惰性 dataset adapter 避免为完整 5M reference 在内存中
一次性创建数百万个 Python label dictionary。
"""

from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import faiss
import h5py
import hydra
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from bioscanclip.epoch.inference_epoch import iter_feature_and_label_batches
from bioscanclip.model.simple_clip import initialize_model_and_load_from_checkpoint
from bioscanclip.util.util import LEVELS, handle_local_ckpt_path


MANIFEST_COLUMNS = [
    "source_group",
    "source_row",
    "processid",
    "sampleid",
    "image_file",
    "dna_bin",
    "order",
    "family",
    "genus",
    "species",
]
MANIFEST_SCHEMA = pa.schema(
    [
        ("source_group", pa.string()),
        ("source_row", pa.int64()),
        ("processid", pa.string()),
        ("sampleid", pa.string()),
        ("image_file", pa.string()),
        ("dna_bin", pa.string()),
        ("order", pa.string()),
        ("family", pa.string()),
        ("genus", pa.string()),
        ("species", pa.string()),
    ]
)
PAIR_MODALITIES = {
    "image_to_image": ("image", "image"),
    "dna_to_dna": ("dna", "dna"),
    "image_to_dna": ("image", "dna"),
}


def _decode(value) -> str:
    """将 HDF5 返回的 bytes 或其他标量统一转换为 UTF-8 字符串。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_strings(dataset: h5py.Dataset, start: int, stop: int) -> np.ndarray:
    """按行区间读取 HDF5 字符串 dataset，并返回 NumPy object array。"""
    return np.asarray(dataset.asstr()[start:stop], dtype=object)


def _write_json(path: Path, value: Mapping) -> None:
    """先写临时文件再原子替换，避免中断后留下不完整 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping:
    """读取 UTF-8 JSON 文件并返回其中的 mapping。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _artifact_root(cfg: DictConfig) -> Path:
    """根据完整/测试规模及 reference set 决定本次运行的 artifact 根目录。"""
    query_limit = cfg.limits.query
    reference_limit = cfg.limits.reference
    if query_limit is None and reference_limit is None:
        namespace = "full"
    else:
        namespace = f"smoke_q{query_limit or 'all'}_r{reference_limit or 'all'}"
    return Path(cfg.output_dir) / namespace / str(cfg.reference_set)


def _query_embedding_path(cfg: DictConfig) -> Path:
    """返回可跨不同 reference set 复用的 query embedding 路径。"""
    query_limit = cfg.limits.query
    namespace = "full" if query_limit is None else f"smoke_q{query_limit}"
    return Path(cfg.output_dir) / namespace / "shared" / "embeddings" / "query.hdf5"


def _manifest_paths(cfg: DictConfig) -> Tuple[Path, Path]:
    """返回当前配置对应的 query manifest 和 reference manifest 路径。"""
    manifest_dir = Path(cfg.output_dir) / "manifests"
    query_path = manifest_dir / "cbg_species_labeled.parquet"
    reference_path = manifest_dir / f"bioscan5m_{cfg.reference_set}.parquet"
    return query_path, reference_path


def _manifest_metadata_path(manifest_path: Path) -> Path:
    """返回某个 Parquet manifest 对应的 JSON metadata 路径。"""
    return manifest_path.with_suffix(".metadata.json")


def _write_manifest_batches(path: Path, batches: Iterable[Mapping[str, Sequence]]) -> int:
    """将多批 manifest 数据增量写入 Parquet，并返回总行数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    writer = pq.ParquetWriter(temporary, MANIFEST_SCHEMA, compression="zstd")
    count = 0
    try:
        for columns in batches:
            table = pa.Table.from_pydict(columns, schema=MANIFEST_SCHEMA)
            if len(table) == 0:
                continue
            writer.write_table(table)
            count += len(table)
    finally:
        writer.close()
    os.replace(temporary, path)
    return count


def _query_manifest_batches(
    hdf5_path: Path,
    group_name: str,
    require_species: bool,
    blank_labels: Sequence[str],
    chunk_size: int = 100_000,
) -> Iterator[Mapping[str, Sequence]]:
    """分块扫描 CBG HDF5，产出符合 query 条件的 manifest 数据批次。"""
    blank = {str(value).strip().lower() for value in blank_labels}
    with h5py.File(hdf5_path, "r", libver="latest") as handle:
        group = handle[group_name]
        size = len(group["processid"])
        if "marker_code" in group:
            marker_values = set(_read_strings(group["marker_code"], 0, size))
            if marker_values != {"COI-5P"}:
                raise ValueError(
                    f"CBG query HDF5 is not COI-5P-only: marker_code={marker_values}"
                )

        for start in range(0, size, chunk_size):
            stop = min(start + chunk_size, size)
            species = _read_strings(group["species"], start, stop)
            if require_species:
                keep = np.asarray(
                    [value.strip().lower() not in blank for value in species],
                    dtype=bool,
                )
            else:
                keep = np.ones(stop - start, dtype=bool)
            rows = np.arange(start, stop, dtype=np.int64)[keep]
            if not len(rows):
                continue
            yield {
                "source_group": [group_name] * len(rows),
                "source_row": rows,
                "processid": _read_strings(group["processid"], start, stop)[keep],
                "sampleid": _read_strings(group["sampleid"], start, stop)[keep],
                "image_file": _read_strings(group["image_file"], start, stop)[keep],
                "dna_bin": _read_strings(group["dna_bin"], start, stop)[keep],
                "order": _read_strings(group["order"], start, stop)[keep],
                "family": _read_strings(group["family"], start, stop)[keep],
                "genus": _read_strings(group["genus"], start, stop)[keep],
                "species": species[keep],
            }


def _reference_manifest_batches(
    hdf5_path: Path,
    groups: Sequence[str],
    deduplicate_processid: bool,
    chunk_size: int = 100_000,
) -> Iterator[Mapping[str, Sequence]]:
    """按 group 优先级扫描 BIOSCAN-5M，并按 processid 选择性去重。"""
    seen_processids = set()
    with h5py.File(hdf5_path, "r", libver="latest") as handle:
        missing = [group for group in groups if group not in handle]
        if missing:
            raise KeyError(f"Missing BIOSCAN-5M HDF5 groups: {missing}")

        for group_name in groups:
            group = handle[group_name]
            size = len(group["processid"])
            for start in range(0, size, chunk_size):
                stop = min(start + chunk_size, size)
                processids = _read_strings(group["processid"], start, stop)
                if deduplicate_processid:
                    keep_values = []
                    for processid in processids:
                        if processid in seen_processids:
                            keep_values.append(False)
                        else:
                            seen_processids.add(processid)
                            keep_values.append(True)
                    keep = np.asarray(keep_values, dtype=bool)
                else:
                    keep = np.ones(stop - start, dtype=bool)
                rows = np.arange(start, stop, dtype=np.int64)[keep]
                if not len(rows):
                    continue
                yield {
                    "source_group": [group_name] * len(rows),
                    "source_row": rows,
                    "processid": processids[keep],
                    "sampleid": _read_strings(group["sampleid"], start, stop)[keep],
                    "image_file": _read_strings(group["image_file"], start, stop)[keep],
                    "dna_bin": _read_strings(group["dna_bin"], start, stop)[keep],
                    "order": _read_strings(group["order"], start, stop)[keep],
                    "family": _read_strings(group["family"], start, stop)[keep],
                    "genus": _read_strings(group["genus"], start, stop)[keep],
                    "species": _read_strings(group["species"], start, stop)[keep],
                }


def prepare_manifests(cfg: DictConfig) -> Tuple[Path, Path]:
    """创建或复用 query/reference manifest，并保存其来源 metadata。"""
    query_path, reference_path = _manifest_paths(cfg)
    overwrite = bool(cfg.overwrite.manifests)
    blank_labels = [str(value) for value in cfg.blank_labels]

    if overwrite or not query_path.exists():
        count = _write_manifest_batches(
            query_path,
            _query_manifest_batches(
                Path(cfg.query_hdf5),
                str(cfg.query_group),
                bool(cfg.query_require_species),
                blank_labels,
            ),
        )
        _write_json(
            _manifest_metadata_path(query_path),
            {
                "kind": "query",
                "hdf5": str(cfg.query_hdf5),
                "group": str(cfg.query_group),
                "query_require_species": bool(cfg.query_require_species),
                "rows": count,
            },
        )
        print(f"Prepared query manifest: {query_path} ({count:,} rows)")
    else:
        print(f"Reusing query manifest: {query_path}")

    reference_cfg = cfg.reference_sets[str(cfg.reference_set)]
    if overwrite or not reference_path.exists():
        count = _write_manifest_batches(
            reference_path,
            _reference_manifest_batches(
                Path(cfg.reference_hdf5),
                [str(value) for value in reference_cfg.groups],
                bool(reference_cfg.deduplicate_processid),
            ),
        )
        _write_json(
            _manifest_metadata_path(reference_path),
            {
                "kind": "reference",
                "reference_set": str(cfg.reference_set),
                "hdf5": str(cfg.reference_hdf5),
                "groups": [str(value) for value in reference_cfg.groups],
                "deduplicate_processid": bool(reference_cfg.deduplicate_processid),
                "rows": count,
            },
        )
        print(f"Prepared reference manifest: {reference_path} ({count:,} rows)")
    else:
        print(f"Reusing reference manifest: {reference_path}")

    return query_path, reference_path


class LazyManifestHDF5Dataset(Dataset):
    """根据 manifest 惰性读取 HDF5，避免将完整 5M 数据集载入内存。"""

    def __init__(self, hdf5_path: Path, manifest_path: Path, limit: Optional[int]):
        """载入轻量行号映射，并构建与原 CLIBD eval 一致的图像变换。"""
        table = pq.read_table(manifest_path, columns=["source_group", "source_row"])
        if limit is not None:
            table = table.slice(0, min(int(limit), len(table)))
        encoded_groups = pc.dictionary_encode(table["source_group"].combine_chunks())
        self.group_codes = encoded_groups.indices.to_numpy(zero_copy_only=False)
        self.group_names = encoded_groups.dictionary.to_pylist()
        self.source_rows = table["source_row"].combine_chunks().to_numpy(
            zero_copy_only=False
        )
        self.hdf5_path = str(hdf5_path)
        self._hdf5 = None
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=256, antialias=True),
                transforms.CenterCrop(224),
            ]
        )

    def __len__(self) -> int:
        """返回当前 manifest（应用可选 limit 后）的样本数量。"""
        return len(self.source_rows)

    def __getstate__(self):
        """供 DataLoader worker pickle 使用，防止跨进程共享 HDF5 handle。"""
        state = self.__dict__.copy()
        state["_hdf5"] = None
        return state

    def _handle(self) -> h5py.File:
        """在当前 worker 首次访问时打开并缓存只读 HDF5 handle。"""
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self.hdf5_path, "r", libver="latest")
        return self._hdf5

    def __getitem__(self, manifest_row: int):
        """读取一行图像、DNA、ID 和 taxonomy，并返回旧 CLIBD 七元组格式。"""
        group_name = self.group_names[int(self.group_codes[manifest_row])]
        source_row = int(self.source_rows[manifest_row])
        group = self._handle()[group_name]

        image_length = int(group["image_mask"][source_row])
        image_bytes = group["image"][source_row][:image_length].tobytes()
        with Image.open(io.BytesIO(image_bytes)) as image:
            image_tensor = self.transform(image.convert("RGB"))

        barcode = _decode(group["barcode"][source_row])
        processid = _decode(group["processid"][source_row])
        labels = {level: _decode(group[level][source_row]) for level in LEVELS}
        # Language is intentionally not encoded for this retrieval task.  These
        # placeholders preserve the established seven-item CLIBD batch contract.
        placeholder = torch.zeros(20, dtype=torch.long)
        return (
            processid,
            image_tensor,
            barcode,
            placeholder,
            placeholder,
            placeholder,
            labels,
        )


def _embedding_path(root: Path, kind: str) -> Path:
    """返回指定 artifact root 下 query/reference embedding 的 HDF5 路径。"""
    return root / "embeddings" / f"{kind}.hdf5"


def _embedding_cache_compatible(
    path: Path,
    expected_rows: int,
    manifest_path: Path,
    checkpoint_path: Path,
    model_output_name: str,
) -> bool:
    """检查已有 cache 的 shape 和来源是否匹配；允许 rows_written 尚未完成。"""
    if not path.exists():
        return False
    with h5py.File(path, "r") as handle:
        return (
            len(handle["image"]) == expected_rows
            and len(handle["dna"]) == expected_rows
            and handle.attrs.get("manifest", "") == str(manifest_path)
            and handle.attrs.get("checkpoint", "") == str(checkpoint_path)
            and handle.attrs.get("model_output_name", "") == model_output_name
            and bool(handle.attrs.get("cache_id", ""))
        )


def _embedding_complete(
    path: Path,
    expected_rows: int,
    manifest_path: Path,
    checkpoint_path: Path,
    model_output_name: str,
) -> bool:
    """检查兼容 cache 是否已经写完所有 embedding rows。"""
    if not _embedding_cache_compatible(
        path,
        expected_rows,
        manifest_path,
        checkpoint_path,
        model_output_name,
    ):
        return False
    with h5py.File(path, "r") as handle:
        return int(handle.attrs.get("rows_written", 0)) == expected_rows


def _open_embedding_cache(
    path: Path,
    rows: int,
    dimension: int,
    metadata: Mapping,
    overwrite: bool,
) -> Tuple[h5py.File, int]:
    """创建或打开可恢复写入的 embedding HDF5，并返回已完成行数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    if path.exists():
        handle = h5py.File(path, "r+")
        if len(handle["image"]) != rows or handle["image"].shape[1] != dimension:
            handle.close()
            raise ValueError(f"Embedding cache shape does not match this run: {path}")
        return handle, int(handle.attrs.get("rows_written", 0))

    handle = h5py.File(path, "w")
    chunk_rows = max(1, min(4096, rows))
    for modality in ("image", "dna"):
        handle.create_dataset(
            modality,
            shape=(rows, dimension),
            dtype=np.float32,
            chunks=(chunk_rows, dimension),
        )
    handle.attrs["rows_written"] = 0
    handle.attrs["l2_normalized"] = True
    for key, value in metadata.items():
        handle.attrs[key] = value
    handle.flush()
    return handle, 0


def _extract_embedding_for_dataset(
    args: DictConfig,
    cfg: DictConfig,
    model,
    device: torch.device,
    kind: str,
    hdf5_path: Path,
    manifest_path: Path,
    limit: Optional[int],
    output_path: Path,
    checkpoint_path: Path,
) -> None:
    """以 DataLoader batch 为单位提取一个完整 dataset 的 image/DNA embedding。

    这里的“一次”指一个 query 或 reference dataset，并不是一个 sample。
    DataLoader 会按照 ``cfg.embedding.batch_size`` 产生 batch，再由 streaming
    inference 接口一次处理整个 batch。
    """
    dataset = LazyManifestHDF5Dataset(hdf5_path, manifest_path, limit)
    rows = len(dataset)
    cache_is_compatible = _embedding_cache_compatible(
        output_path,
        rows,
        manifest_path,
        checkpoint_path,
        str(args.model_config.model_output_name),
    )
    cache_is_valid = _embedding_complete(
        output_path,
        rows,
        manifest_path,
        checkpoint_path,
        str(args.model_config.model_output_name),
    )
    if cache_is_valid and not bool(cfg.overwrite.embeddings):
        print(f"Reusing complete {kind} embeddings: {output_path}")
        return
    if output_path.exists() and not cache_is_compatible:
        print(f"Discarding incompatible {kind} embedding cache: {output_path}")
        output_path.unlink()
    cache, start = _open_embedding_cache(
        output_path,
        rows,
        int(args.model_config.output_dim),
        {
            "kind": kind,
            "manifest": str(manifest_path),
            "checkpoint": str(checkpoint_path),
            "model_output_name": str(args.model_config.model_output_name),
            # cache_id 只用于识别 embedding 是否被重新生成，不计算大文件 hash。
            "cache_id": str(uuid4()),
        },
        bool(cfg.overwrite.embeddings),
    )
    OmegaConf.save(
        config=args,
        f=output_path.parent / "config.yaml",
        resolve=True,
    )
    if start >= rows:
        cache.close()
        return

    subset = Subset(dataset, range(start, rows))
    loader = DataLoader(
        subset,
        batch_size=int(cfg.embedding.batch_size),
        shuffle=False,
        num_workers=int(cfg.embedding.num_workers),
        pin_memory=bool(cfg.embedding.pin_memory),
    )
    cursor = start
    try:
        for batch_number, output in enumerate(
            iter_feature_and_label_batches(
                loader,
                model,
                device,
                encode_language=False,
            ),
            start=1,
        ):
            image = output["encoded_image_feature"]
            dna = output["encoded_dna_feature"]
            batch_rows = len(output["file_name_list"])
            if image is None or dna is None:
                raise RuntimeError("The selected CLIBD checkpoint must encode image and DNA")
            cache["image"][cursor:cursor + batch_rows] = image.astype(np.float32)
            cache["dna"][cursor:cursor + batch_rows] = dna.astype(np.float32)
            cursor += batch_rows
            cache.attrs.modify("rows_written", cursor)
            if batch_number % int(cfg.embedding.flush_every_batches) == 0:
                cache.flush()
        cache.flush()
    finally:
        cache.close()
    print(f"Completed {kind} embeddings: {output_path} ({cursor:,} rows)")


def extract_embeddings(args: DictConfig, cfg: DictConfig, query_path: Path, reference_path: Path) -> None:
    """加载一次 CLIBD checkpoint，依次生成或复用 query/reference embedding。"""
    root = _artifact_root(cfg)
    query_table_rows = pq.read_metadata(query_path).num_rows
    reference_table_rows = pq.read_metadata(reference_path).num_rows
    query_rows = min(query_table_rows, int(cfg.limits.query)) if cfg.limits.query else query_table_rows
    reference_rows = (
        min(reference_table_rows, int(cfg.limits.reference))
        if cfg.limits.reference
        else reference_table_rows
    )
    query_cache = _query_embedding_path(cfg)
    reference_cache = _embedding_path(root, "reference")
    checkpoint_path = Path(handle_local_ckpt_path(args))
    if checkpoint_path != Path(cfg.checkpoint_path):
        raise ValueError(
            "Configured retrieval checkpoint does not match the checkpoint "
            f"selected by CLIBD: {cfg.checkpoint_path} != {checkpoint_path}"
        )
    if (
        _embedding_complete(
            query_cache,
            query_rows,
            query_path,
            checkpoint_path,
            str(args.model_config.model_output_name),
        )
        and _embedding_complete(
            reference_cache,
            reference_rows,
            reference_path,
            checkpoint_path,
            str(args.model_config.model_output_name),
        )
        and not bool(cfg.overwrite.embeddings)
    ):
        for cache_path in (query_cache, reference_cache):
            config_path = cache_path.parent / "config.yaml"
            if not config_path.exists():
                OmegaConf.save(config=args, f=config_path, resolve=True)
        print("Reusing complete query and reference embedding caches")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model_and_load_from_checkpoint(args, device=device).to(device)
    model.eval()
    _extract_embedding_for_dataset(
        args,
        cfg,
        model,
        device,
        "query",
        Path(cfg.query_hdf5),
        query_path,
        cfg.limits.query,
        query_cache,
        checkpoint_path,
    )
    _extract_embedding_for_dataset(
        args,
        cfg,
        model,
        device,
        "reference",
        Path(cfg.reference_hdf5),
        reference_path,
        cfg.limits.reference,
        reference_cache,
        checkpoint_path,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _index_path(root: Path, modality: str) -> Path:
    """返回指定 modality 对应的 FAISS index 文件路径。"""
    return root / "indices" / f"{modality}.faiss"


def _iter_hdf5_rows(dataset: h5py.Dataset, batch_size: int):
    """顺序分批读取二维 HDF5 dataset，产出起止行号和 float32 数据。"""
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        yield start, stop, np.asarray(dataset[start:stop], dtype=np.float32)


def _sample_hdf5_rows(
    dataset: h5py.Dataset,
    sample_size: int,
    seed: int,
    read_batch_size: int,
) -> np.ndarray:
    """均匀抽样 embedding，同时保持 HDF5 顺序读取以提高 I/O 效率。"""
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(dataset), sample_size, replace=False))
    samples = []
    cursor = 0
    for start, stop, batch in _iter_hdf5_rows(dataset, read_batch_size):
        next_cursor = np.searchsorted(selected, stop, side="left")
        local_rows = selected[cursor:next_cursor] - start
        if len(local_rows):
            samples.append(batch[local_rows])
        cursor = next_cursor
    return np.concatenate(samples, axis=0)


def build_indices(cfg: DictConfig) -> None:
    """为检索所需的 reference modality 创建或复用 Flat/IVF-Flat index。"""
    root = _artifact_root(cfg)
    embedding_path = _embedding_path(root, "reference")
    if not embedding_path.exists():
        raise FileNotFoundError(f"Reference embedding cache is missing: {embedding_path}")
    reference_cfg = cfg.reference_sets[str(cfg.reference_set)]
    index_type = str(reference_cfg.index_type)
    required_modalities = sorted(
        {PAIR_MODALITIES[str(pair)][1] for pair in cfg.retrieval.pairs}
    )

    with h5py.File(embedding_path, "r") as embeddings:
        embedding_cache_id = str(embeddings.attrs["cache_id"])
        for modality in required_modalities:
            output_path = _index_path(root, modality)
            metadata_path = output_path.with_suffix(".metadata.json")
            if index_type == "ivf_flat":
                index_parameters = {
                    "nlist": min(
                        int(cfg.index.ivf_nlist),
                        max(1, len(embeddings[modality]) // 40),
                    ),
                    "train_size": min(
                        int(cfg.index.ivf_train_size), len(embeddings[modality])
                    ),
                    "random_seed": int(cfg.index.random_seed),
                }
            else:
                index_parameters = {}
            expected_metadata = {
                "modality": modality,
                "index_type": index_type,
                "rows": len(embeddings[modality]),
                "dimension": embeddings[modality].shape[1],
                "embedding_cache": str(embedding_path),
                "embedding_cache_id": embedding_cache_id,
                "index_parameters": index_parameters,
            }
            reusable = (
                output_path.exists()
                and metadata_path.exists()
                and _read_json(metadata_path) == expected_metadata
            )
            if reusable and not bool(cfg.overwrite.indices):
                print(f"Reusing FAISS index: {output_path}")
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            values = embeddings[modality]
            dimension = values.shape[1]
            if index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif index_type == "ivf_flat":
                nlist = index_parameters["nlist"]
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(
                    quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
                )
                train_size = index_parameters["train_size"]
                training = _sample_hdf5_rows(
                    values,
                    train_size,
                    int(cfg.index.random_seed),
                    int(cfg.index.add_batch_size),
                )
                print(
                    f"Training IVF index for {modality}: "
                    f"nlist={nlist:,}, samples={train_size:,}"
                )
                index.train(training)
            else:
                raise ValueError(f"Unsupported index_type: {index_type}")

            for start, stop, batch in _iter_hdf5_rows(
                values, int(cfg.index.add_batch_size)
            ):
                index.add(batch)
                if start == 0 or stop == len(values) or stop % 1_000_000 == 0:
                    print(f"Indexing {modality}: {stop:,}/{len(values):,}")
            temporary = output_path.with_suffix(".faiss.partial")
            faiss.write_index(index, str(temporary))
            os.replace(temporary, output_path)
            _write_json(metadata_path, expected_metadata)
            print(f"Saved FAISS index: {output_path}")


def _limited_manifest_table(path: Path, limit: Optional[int]) -> pa.Table:
    """读取 manifest 的检索所需列，并在 smoke test 时截取前 N 行。"""
    table = pq.read_table(path, columns=MANIFEST_COLUMNS)
    if limit is not None:
        table = table.slice(0, min(int(limit), len(table)))
    return table


def _prediction_path(root: Path, pair: str) -> Path:
    """返回某个 query→key modality pair 的 Top-K Parquet 输出路径。"""
    return root / "predictions" / f"{pair}.parquet"


def search(cfg: DictConfig, query_manifest: Path, reference_manifest: Path) -> None:
    """分批执行 FAISS Top-K 检索，并保存相似度、ID、来源和 taxonomy。"""
    root = _artifact_root(cfg)
    reference_cfg = cfg.reference_sets[str(cfg.reference_set)]
    search_device = str(reference_cfg.search_device)
    query_embeddings_path = _query_embedding_path(cfg)
    if not query_embeddings_path.exists():
        raise FileNotFoundError(f"Query embedding cache is missing: {query_embeddings_path}")
    query_table = _limited_manifest_table(query_manifest, cfg.limits.query)
    reference_table = _limited_manifest_table(reference_manifest, cfg.limits.reference)
    max_k = max(int(value) for value in cfg.retrieval.top_k)

    with h5py.File(query_embeddings_path, "r") as query_embeddings:
        query_cache_id = str(query_embeddings.attrs["cache_id"])
        for pair_value in cfg.retrieval.pairs:
            pair = str(pair_value)
            query_modality, key_modality = PAIR_MODALITIES[pair]
            output_path = _prediction_path(root, pair)
            index_metadata_path = _index_path(root, key_modality).with_suffix(
                ".metadata.json"
            )
            index_metadata = _read_json(index_metadata_path)
            prediction_metadata = {
                "pair": pair,
                "max_k": max_k,
                "query_cache_id": query_cache_id,
                "index_metadata": index_metadata,
                "ivf_nprobe": int(cfg.index.ivf_nprobe),
                "search_device": search_device,
            }
            prediction_metadata_path = output_path.with_suffix(".metadata.json")
            reusable = (
                output_path.exists()
                and prediction_metadata_path.exists()
                and _read_json(prediction_metadata_path) == prediction_metadata
            )
            if reusable and not bool(cfg.overwrite.predictions):
                print(f"Reusing predictions: {output_path}")
                continue

            index = faiss.read_index(str(_index_path(root, key_modality)))
            if isinstance(index, faiss.IndexIVF):
                index.nprobe = int(cfg.index.ivf_nprobe)
            gpu_resources = None
            if search_device == "gpu":
                if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() < 1:
                    raise RuntimeError("R0 is configured for FAISS GPU search, but no FAISS GPU is available")
                gpu_resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
                print(f"Searching {pair} with FAISS GPU")
            elif search_device != "cpu":
                raise ValueError(f"Unsupported FAISS search_device: {search_device}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".parquet.partial")
            writer = None
            try:
                values = query_embeddings[query_modality]
                batch_size = int(cfg.retrieval.query_batch_size)
                for start, stop, query_batch in _iter_hdf5_rows(values, batch_size):
                    similarities, indices = index.search(query_batch, max_k)
                    flat_indices = indices.reshape(-1)
                    if np.any(flat_indices < 0):
                        raise RuntimeError("FAISS returned an invalid neighbor index")
                    keys = reference_table.take(pa.array(flat_indices))
                    query_rows = np.repeat(np.arange(start, stop, dtype=np.int64), max_k)
                    queries = query_table.take(pa.array(query_rows))
                    columns = {
                        "query_row": query_rows,
                        "rank": np.tile(
                            np.arange(1, max_k + 1, dtype=np.int16), stop - start
                        ),
                        "key_row": flat_indices.astype(np.int64),
                        "similarity": similarities.reshape(-1).astype(np.float32),
                        "query_processid": queries["processid"],
                        "key_processid": keys["processid"],
                        "query_sampleid": queries["sampleid"],
                        "key_sampleid": keys["sampleid"],
                        "query_source_group": queries["source_group"],
                        "query_source_row": queries["source_row"],
                        "key_source_group": keys["source_group"],
                        "key_source_row": keys["source_row"],
                        "query_dna_bin": queries["dna_bin"],
                        "key_dna_bin": keys["dna_bin"],
                    }
                    for level in LEVELS:
                        columns[f"query_{level}"] = queries[level]
                        columns[f"key_{level}"] = keys[level]
                    table = pa.Table.from_pydict(columns)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary, table.schema, compression="zstd"
                        )
                    writer.write_table(table)
                    print(f"Searching {pair}: {stop:,}/{len(values):,}")
            finally:
                if writer is not None:
                    writer.close()
            if writer is None:
                raise RuntimeError("No queries were written")
            os.replace(temporary, output_path)
            _write_json(prediction_metadata_path, prediction_metadata)
            print(f"Saved predictions: {output_path}")


class StreamingAccuracy:
    """以有限内存累计与原 CLIBD 定义等价的 Micro/Macro Top-K 指标。"""

    def __init__(self, k_values: Sequence[int], blank_labels: Sequence[str], exclude_blank: bool):
        """初始化各 taxonomy rank、K 值和 class-level 计数器。"""
        self.k_values = sorted(int(value) for value in k_values)
        self.blank_labels = {str(value).strip().lower() for value in blank_labels}
        self.exclude_blank = exclude_blank
        self.total = defaultdict(int)
        self.correct = {k: defaultdict(int) for k in self.k_values}
        self.class_total = {level: defaultdict(int) for level in LEVELS}
        self.class_correct = {
            k: {level: defaultdict(int) for level in LEVELS}
            for k in self.k_values
        }

    def add(self, ground_truth: Mapping[str, str], predictions: Mapping[str, Sequence[str]]) -> None:
        """加入一个 query 的真值和有序预测，更新 Micro/Macro 所需计数。"""
        for level in LEVELS:
            label = ground_truth[level]
            if self.exclude_blank and label.strip().lower() in self.blank_labels:
                continue
            self.total[level] += 1
            self.class_total[level][label] += 1
            for k in self.k_values:
                if label in predictions[level][:k]:
                    self.correct[k][level] += 1
                    self.class_correct[k][level][label] += 1

    def result(self) -> Mapping:
        """将累计计数转换为 Micro、Macro 和 per-class accuracy。"""
        micro = {}
        macro = {}
        per_class = {}
        for k in self.k_values:
            micro[str(k)] = {}
            macro[str(k)] = {}
            per_class[str(k)] = {}
            for level in LEVELS:
                denominator = self.total[level]
                micro[str(k)][level] = (
                    self.correct[k][level] / denominator if denominator else None
                )
                class_values = {
                    label: self.class_correct[k][level][label] / count
                    for label, count in self.class_total[level].items()
                }
                macro[str(k)][level] = (
                    float(np.mean(list(class_values.values()))) if class_values else None
                )
                per_class[str(k)][level] = class_values
        return {
            "num_queries_by_rank": dict(self.total),
            "micro_accuracy": micro,
            "macro_accuracy": macro,
            "per_class_accuracy": per_class,
        }


def evaluate_prediction_file(cfg: DictConfig, prediction_path: Path) -> Mapping:
    """流式读取一个 prediction Parquet，并计算各 rank 的 Top-K 指标。"""
    metric = StreamingAccuracy(
        cfg.retrieval.top_k,
        cfg.blank_labels,
        bool(cfg.retrieval.exclude_blank_ground_truth),
    )
    current_query = None
    ground_truth = None
    predictions = None

    def finish_query():
        """在 query 边界处将上一条 query 的完整 Top-K 结果加入指标。"""
        if ground_truth is not None:
            metric.add(ground_truth, predictions)

    parquet = pq.ParquetFile(prediction_path)
    columns = ["query_row"] + [
        name for level in LEVELS for name in (f"query_{level}", f"key_{level}")
    ]
    for record_batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        values = record_batch.to_pydict()
        for row_number, query_row in enumerate(values["query_row"]):
            if query_row != current_query:
                finish_query()
                current_query = query_row
                ground_truth = {
                    level: values[f"query_{level}"][row_number] for level in LEVELS
                }
                predictions = {level: [] for level in LEVELS}
            for level in LEVELS:
                predictions[level].append(values[f"key_{level}"][row_number])
    finish_query()
    return metric.result()


def evaluate(args: DictConfig, cfg: DictConfig) -> None:
    """评估全部 modality pair，并在 result 旁保存指标及完整 resolved config。"""
    root = _artifact_root(cfg)
    all_results = {
        "reference_set": str(cfg.reference_set),
        "top_k": [int(value) for value in cfg.retrieval.top_k],
        "pairs": {},
    }
    for pair_value in cfg.retrieval.pairs:
        pair = str(pair_value)
        prediction_path = _prediction_path(root, pair)
        if not prediction_path.exists():
            raise FileNotFoundError(f"Prediction file is missing: {prediction_path}")
        result = evaluate_prediction_file(cfg, prediction_path)
        all_results["pairs"][pair] = result
        print(
            f"{pair}: species Micro@1="
            f"{result['micro_accuracy']['1']['species']:.4f}, "
            f"Macro@1={result['macro_accuracy']['1']['species']:.4f}"
        )
    output_path = root / "metrics" / "retrieval_metrics.json"
    _write_json(output_path, all_results)
    OmegaConf.save(
        config=args,
        f=root / "config.yaml",
        resolve=True,
    )
    print(f"Saved retrieval metrics: {output_path}")


@hydra.main(
    config_path="../../bioscanclip/config",
    config_name="cbg_retrieval",
    version_base="1.3",
)
def main(args: DictConfig) -> None:
    """Hydra 入口：按配置依次或单独执行五个 retrieval stage。"""
    cfg = args.cbg_retrieval
    stage = str(cfg.stage)
    valid_stages = {"prepare", "embed", "index", "search", "evaluate", "all"}
    if stage not in valid_stages:
        raise ValueError(f"stage must be one of {sorted(valid_stages)}, got {stage}")

    query_manifest, reference_manifest = _manifest_paths(cfg)
    if stage in {"prepare", "all"}:
        query_manifest, reference_manifest = prepare_manifests(cfg)
    elif not query_manifest.exists() or not reference_manifest.exists():
        raise FileNotFoundError("Run cbg_retrieval.stage=prepare first")

    if stage in {"embed", "all"}:
        extract_embeddings(args, cfg, query_manifest, reference_manifest)
    if stage in {"index", "all"}:
        build_indices(cfg)
    if stage in {"search", "all"}:
        search(cfg, query_manifest, reference_manifest)
    if stage in {"evaluate", "all"}:
        evaluate(args, cfg)


if __name__ == "__main__":
    main()
