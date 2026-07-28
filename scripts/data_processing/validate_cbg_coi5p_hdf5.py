#!/usr/bin/env python3
"""Validate and inspect a generated CBG COI-5P HDF5 file.

The validator performs full structural and metadata checks, then samples image
rows using the same ``image``/``image_mask`` byte slicing used by CLIBD. Sampled
images can be extracted byte-for-byte to a separate directory for manual
inspection. Neither the HDF5 input nor the resized source images are modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator

import h5py
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GROUP_NAME = "no_split"
MARKER_CODE = "COI-5P"
NOT_CLASSIFIED = "not_classified"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TAXONOMY_FIELDS = ("order", "family", "genus", "species")
METADATA_FIELDS = (
    "barcode",
    "dna_bin",
    "order",
    "family",
    "genus",
    "species",
    "processid",
    "sampleid",
    "image_file",
    "source_image_file",
    "record_id",
    "specimenid",
    "marker_code",
)
REQUIRED_DATASETS = set(METADATA_FIELDS) | {"image", "image_mask"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a CBG COI-5P HDF5 file and optionally extract sampled "
            "images and metadata for manual inspection."
        )
    )
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--group", default=GROUP_NAME)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--resized-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--metadata-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--skip-full-metadata-check",
        action="store_true",
        help="skip row-by-row comparison with the filtered source CSV",
    )
    parser.add_argument(
        "--skip-source-image-comparison",
        action="store_true",
        help="decode sampled HDF5 images but do not compare them to source PNGs",
    )
    parser.add_argument(
        "--skip-clibd-input-check",
        action="store_true",
        help="skip CLIBD 224x224 image transform and DNA tokenizer checks",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        help="write sampled PNGs and extracted_records.csv to this new directory",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path; defaults beside the HDF5 file",
    )
    return parser.parse_args()


def decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def normalized_taxonomy(value: str) -> str:
    value = value.strip()
    return value if value else NOT_CLASSIFIED


def expected_record(row: dict[str, str]) -> dict[str, str]:
    source_image_file = row["image_file"].strip()
    dna_bin = row["dna_bin"].strip() or NOT_CLASSIFIED
    return {
        "barcode": row["dna_barcode"].strip(),
        "dna_bin": dna_bin,
        "order": normalized_taxonomy(row["order"]),
        "family": normalized_taxonomy(row["family"]),
        "genus": normalized_taxonomy(row["genus"]),
        "species": normalized_taxonomy(row["species"]),
        "processid": row["processid"].strip(),
        "sampleid": row["sampleid"].strip(),
        "image_file": f"{Path(source_image_file).stem}.png",
        "source_image_file": source_image_file,
        "record_id": row["record_id"].strip(),
        "specimenid": row["specimenid"].strip(),
        "marker_code": MARKER_CODE,
    }


def eligible_metadata_rows(metadata: Path) -> Iterator[dict[str, str]]:
    with metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sampleid",
            "processid",
            "image_file",
            "dna_barcode",
            "dna_bin",
            "order",
            "family",
            "genus",
            "species",
            "record_id",
            "specimenid",
            "marker_code",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
        for row in reader:
            if row["marker_code"].strip() != MARKER_CODE:
                continue
            if not all(
                row[field].strip()
                for field in ("sampleid", "processid", "image_file", "dna_barcode")
            ):
                continue
            yield expected_record(row)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sample_indices(count: int, requested: int, seed: int) -> list[int]:
    if requested >= count:
        return list(range(count))
    if requested == 1:
        return [0]
    rng = random.Random(seed)
    fixed = {0, count - 1}
    remaining = requested - len(fixed)
    if remaining > 0:
        candidates = range(1, count - 1)
        fixed.update(rng.sample(candidates, remaining))
    return sorted(fixed)


def validate_structure(
    file: h5py.File, group_name: str
) -> tuple[h5py.Group, int, dict[str, Any]]:
    if group_name not in file:
        raise ValueError(
            f"Missing group {group_name!r}; available groups: {list(file.keys())}"
        )
    if len(file.keys()) != 1:
        raise ValueError(f"Expected one HDF5 group, found: {list(file.keys())}")
    group = file[group_name]
    missing = REQUIRED_DATASETS - set(group.keys())
    if missing:
        raise ValueError(f"Missing HDF5 datasets: {sorted(missing)}")

    count = len(group["image"])
    if count <= 0:
        raise ValueError("HDF5 contains no records")
    for name in REQUIRED_DATASETS:
        if len(group[name]) != count:
            raise ValueError(
                f"Dataset {name} has {len(group[name])} rows; expected {count}"
            )
    if group["image"].ndim != 2 or group["image"].dtype != np.uint8:
        raise ValueError(
            f"image must be 2D uint8, got {group['image'].shape} "
            f"{group['image'].dtype}"
        )
    if group["image_mask"].ndim != 1:
        raise ValueError("image_mask must be one-dimensional")
    if "complete" in file.attrs and not bool(file.attrs["complete"]):
        raise ValueError("HDF5 root attribute complete is false")
    if int(group.attrs.get("rows_written", count)) != count:
        raise ValueError("rows_written does not equal the HDF5 record count")

    summary = {
        "groups": list(file.keys()),
        "datasets": sorted(group.keys()),
        "records": count,
        "image_shape": list(group["image"].shape),
        "image_dtype": str(group["image"].dtype),
        "image_chunks": list(group["image"].chunks or []),
        "image_compression": group["image"].compression,
        "root_complete_attribute": bool(file.attrs.get("complete", True)),
    }
    return group, count, summary


def validate_all_masks_and_identifiers(
    group: h5py.Group, count: int, chunk_size: int
) -> dict[str, Any]:
    image_width = group["image"].shape[1]
    seen_sampleids: set[str] = set()
    seen_processids: set[str] = set()
    minimum_length = image_width
    maximum_length = 0
    total_length = 0
    not_classified_counts = {
        field: 0 for field in (*TAXONOMY_FIELDS, "dna_bin")
    }

    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        masks = group["image_mask"][start:end].astype(np.int64)
        if np.any(masks <= 0) or np.any(masks > image_width):
            bad = np.flatnonzero((masks <= 0) | (masks > image_width))[0]
            raise ValueError(
                f"Invalid image_mask at HDF5 row {start + int(bad)}: "
                f"{int(masks[bad])}"
            )
        minimum_length = min(minimum_length, int(masks.min()))
        maximum_length = max(maximum_length, int(masks.max()))
        total_length += int(masks.sum())

        sampleids = [decode_string(v) for v in group["sampleid"][start:end]]
        processids = [decode_string(v) for v in group["processid"][start:end]]
        barcodes = [decode_string(v) for v in group["barcode"][start:end]]
        markers = [decode_string(v) for v in group["marker_code"][start:end]]
        for offset, (sampleid, processid, barcode, marker) in enumerate(
            zip(sampleids, processids, barcodes, markers)
        ):
            row_index = start + offset
            if not sampleid or sampleid in seen_sampleids:
                raise ValueError(
                    f"Blank or duplicate sampleid at HDF5 row {row_index}: "
                    f"{sampleid!r}"
                )
            if not processid or processid in seen_processids:
                raise ValueError(
                    f"Blank or duplicate processid at HDF5 row {row_index}: "
                    f"{processid!r}"
                )
            if not barcode:
                raise ValueError(f"Blank barcode at HDF5 row {row_index}")
            if marker != MARKER_CODE:
                raise ValueError(
                    f"Unexpected marker at HDF5 row {row_index}: {marker!r}"
                )
            seen_sampleids.add(sampleid)
            seen_processids.add(processid)

        for field in not_classified_counts:
            values = [decode_string(v) for v in group[field][start:end]]
            if any(not value for value in values):
                raise ValueError(f"Blank {field} between rows {start} and {end}")
            not_classified_counts[field] += sum(
                value == NOT_CLASSIFIED for value in values
            )

    return {
        "unique_sampleids": len(seen_sampleids),
        "unique_processids": len(seen_processids),
        "minimum_encoded_bytes": minimum_length,
        "maximum_encoded_bytes": maximum_length,
        "total_encoded_bytes": total_length,
        "not_classified_counts": not_classified_counts,
    }


def compare_full_metadata(
    group: h5py.Group,
    metadata: Path,
    count: int,
    chunk_size: int,
    allow_additional_source_rows: bool,
) -> int:
    iterator = eligible_metadata_rows(metadata)
    compared = 0
    while compared < count:
        end = min(compared + chunk_size, count)
        expected_chunk = []
        for _ in range(end - compared):
            try:
                expected_chunk.append(next(iterator))
            except StopIteration as error:
                raise ValueError(
                    f"Metadata ended after {compared + len(expected_chunk)} "
                    f"eligible rows; HDF5 contains {count}"
                ) from error

        for field in METADATA_FIELDS:
            observed = [
                decode_string(value) for value in group[field][compared:end]
            ]
            for offset, (actual, expected) in enumerate(
                zip(observed, expected_chunk)
            ):
                wanted = expected[field]
                if actual != wanted:
                    raise ValueError(
                        f"Metadata mismatch at HDF5 row {compared + offset}, "
                        f"field {field}: HDF5={actual!r}, CSV={wanted!r}"
                    )
        compared = end
        if compared % (chunk_size * 10) == 0 or compared == count:
            print(f"metadata_compared={compared}/{count}", flush=True)

    if not allow_additional_source_rows:
        try:
            extra = next(iterator)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ValueError(
                f"Source metadata has more eligible records than HDF5; next "
                f"sampleid is {extra['sampleid']}"
            )
    return compared


def prepare_extract_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"Extract path exists and is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise FileExistsError(
                f"Refusing to write into non-empty extract directory: {resolved}"
            )
    else:
        resolved.mkdir(parents=True)
    return resolved


def load_clibd_checks() -> tuple[Any, Any]:
    from torchvision import transforms
    from transformers import AutoTokenizer

    transform = transforms.Compose(
        [
            transforms.Resize(size=256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    # Match the tokenizer and parameters used by inference_epoch.py.
    tokenizer = AutoTokenizer.from_pretrained(
        "bioscan-ml/BarcodeBERT", trust_remote_code=True
    )
    return transform, tokenizer


def validate_sampled_images(
    group: h5py.Group,
    indices: list[int],
    resized_dir: Path | None,
    extract_dir: Path | None,
    run_clibd_checks: bool,
) -> tuple[list[dict[str, Any]], str]:
    transform = tokenizer = None
    if run_clibd_checks:
        transform, tokenizer = load_clibd_checks()

    extracted_rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for position, index in enumerate(indices, 1):
        length = int(group["image_mask"][index])
        padded = group["image"][index].astype(np.uint8)
        encoded = padded[:length].tobytes()
        if not encoded.startswith(PNG_SIGNATURE):
            raise ValueError(f"Non-PNG signature at HDF5 row {index}")
        digest.update(encoded)

        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            image_format = image.format
            image_mode = image.mode
            image_width, image_height = image.size
            if image_format != "PNG":
                raise ValueError(
                    f"Unexpected image format at row {index}: {image_format}"
                )
            if image_mode != "RGB":
                raise ValueError(
                    f"Unexpected image mode at row {index}: {image_mode}"
                )
            if min(image_width, image_height) != 256:
                raise ValueError(
                    f"Unexpected image size at row {index}: {image.size}"
                )
            if transform is not None:
                tensor = transform(image)
                if tuple(tensor.shape) != (3, 224, 224):
                    raise ValueError(
                        f"CLIBD transform shape at row {index}: "
                        f"{tuple(tensor.shape)}"
                    )

        row = {"hdf5_index": index}
        for field in METADATA_FIELDS:
            row[field] = decode_string(group[field][index])
        row.update(
            {
                "encoded_bytes": length,
                "decoded_width": image_width,
                "decoded_height": image_height,
                "decoded_mode": image_mode,
                "clibd_tensor_shape": (
                    "3x224x224" if run_clibd_checks else "not_checked"
                ),
            }
        )

        if tokenizer is not None:
            tokenized = tokenizer(
                row["barcode"],
                padding="max_length",
                truncation=True,
                max_length=133,
                return_tensors="pt",
            )
            token_count = int(tokenized["input_ids"].shape[-1])
            if token_count <= 0:
                raise ValueError(
                    f"DNA tokenizer produced no tokens at row {index}"
                )
            row["dna_token_count"] = token_count
        else:
            row["dna_token_count"] = "not_checked"

        if resized_dir is not None:
            source_path = resized_dir / row["image_file"]
            with source_path.open("rb") as handle:
                source_bytes = handle.read()
            if encoded != source_bytes:
                raise ValueError(
                    f"HDF5 bytes differ from resized source at row {index}: "
                    f"{source_path}"
                )
            row["source_bytes_equal"] = True
        else:
            row["source_bytes_equal"] = "not_checked"

        if extract_dir is not None:
            destination = extract_dir / row["image_file"]
            with destination.open("xb") as handle:
                handle.write(encoded)
            row["extracted_path"] = str(destination)
        else:
            row["extracted_path"] = ""
        extracted_rows.append(row)

        if position % 25 == 0 or position == len(indices):
            print(
                f"sampled_images_validated={position}/{len(indices)}", flush=True
            )
    return extracted_rows, digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.metadata_chunk_size <= 0:
        raise ValueError("--metadata-chunk-size must be positive")

    hdf5_path = args.hdf5.resolve(strict=True)
    metadata = args.metadata.resolve(strict=True)
    resized_dir = None
    if not args.skip_source_image_comparison:
        resized_dir = args.resized_dir.resolve(strict=True)
        if not resized_dir.is_dir():
            raise ValueError(f"Resized image path is not a directory: {resized_dir}")
    extract_dir = prepare_extract_dir(args.extract_dir)
    report = (
        args.report.resolve()
        if args.report
        else hdf5_path.with_name(f"{hdf5_path.stem}_validation_report.json")
    )
    report.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    print(f"hdf5={hdf5_path}", flush=True)
    print(f"group={args.group}", flush=True)
    print(f"metadata={metadata}", flush=True)
    print(f"resized_dir={resized_dir}", flush=True)
    print(f"extract_dir={extract_dir}", flush=True)

    with h5py.File(hdf5_path, "r", libver="latest") as file:
        group, count, structure = validate_structure(file, args.group)
        print(f"records={count}", flush=True)
        full_checks = validate_all_masks_and_identifiers(
            group, count, args.metadata_chunk_size
        )
        metadata_compared = 0
        if not args.skip_full_metadata_check:
            source_was_limited = int(file.attrs.get("limit", -1)) >= 0
            metadata_compared = compare_full_metadata(
                group,
                metadata,
                count,
                args.metadata_chunk_size,
                source_was_limited,
            )
        indices = sample_indices(
            count, min(args.sample_count, count), args.seed
        )
        sampled_rows, sampled_digest = validate_sampled_images(
            group,
            indices,
            resized_dir,
            extract_dir,
            not args.skip_clibd_input_check,
        )

    if extract_dir is not None:
        fieldnames = list(sampled_rows[0].keys())
        atomic_write_csv(
            extract_dir / "extracted_records.csv", fieldnames, sampled_rows
        )

    payload = {
        "status": "COMPLETE",
        "hdf5": str(hdf5_path),
        "hdf5_file_bytes": hdf5_path.stat().st_size,
        "group": args.group,
        "metadata": str(metadata),
        "resized_image_dir": str(resized_dir) if resized_dir else None,
        "extract_dir": str(extract_dir) if extract_dir else None,
        "structure": structure,
        "full_checks": full_checks,
        "full_metadata_comparison_enabled": not args.skip_full_metadata_check,
        "metadata_rows_compared": metadata_compared,
        "sample_count": len(sampled_rows),
        "sample_indices": indices,
        "sampled_image_bytes_sha256": sampled_digest,
        "source_image_bytes_compared": resized_dir is not None,
        "clibd_input_checks_enabled": not args.skip_clibd_input_check,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_write_json(report, payload)
    print(f"report={report}", flush=True)
    print("validation_status=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
