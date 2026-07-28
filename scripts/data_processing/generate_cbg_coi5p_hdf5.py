#!/usr/bin/env python3
"""Build a CLIBD-compatible HDF5 file for the CBG COI-5P dataset.

The resized PNG directory is strictly read-only to this program. Images are
copied byte-for-byte into a padded ``uint8`` dataset and their true lengths are
stored in ``image_mask``, matching the access pattern used by CLIBD.

The output is first written to a hidden ``.partial`` HDF5 file. Interrupted
runs can resume from the last flushed batch, and the partial file is renamed to
the requested output only after validation succeeds.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

import h5py
import numpy as np
from PIL import Image


GROUP_NAME = "no_split"
MARKER_CODE = "COI-5P"
NOT_CLASSIFIED = "not_classified"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TAXONOMY_FIELDS = ("order", "family", "genus", "species")
STRING_FIELDS = (
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


@dataclass(frozen=True)
class Record:
    sampleid: str
    processid: str
    barcode: str
    dna_bin: str
    order: str
    family: str
    genus: str
    species: str
    image_file: str
    source_image_file: str
    record_id: str
    specimenid: str
    marker_code: str
    encoded_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a resumable CLIBD-compatible HDF5 file from resized CBG "
            "COI-5P PNG images."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--resize-manifest", type=Path, required=True)
    parser.add_argument("--resized-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--excluded-output",
        type=Path,
        help="CSV for filtered records; defaults beside --output",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="JSON generation report; defaults beside --output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="write only the first N eligible records (smoke tests only)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--read-workers", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument(
        "--compression",
        choices=("lzf", "none"),
        default="lzf",
        help="transparent HDF5 compression for padded image rows",
    )
    parser.add_argument(
        "--restart-partial",
        action="store_true",
        help="delete only this output's incomplete partial file and restart",
    )
    return parser.parse_args()


def normalized_taxonomy(value: str) -> str:
    value = value.strip()
    return value if value else NOT_CLASSIFIED


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]
) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def derived_sidecar(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def partial_path_for(output: Path) -> Path:
    return output.with_name(f".{output.name}.partial")


def path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def validate_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    metadata = args.metadata.resolve(strict=True)
    resize_manifest = args.resize_manifest.resolve(strict=True)
    resized_dir = args.resized_dir.resolve(strict=True)
    if not resized_dir.is_dir():
        raise ValueError(f"Resized image path is not a directory: {resized_dir}")

    output = args.output.resolve()
    excluded_output = (
        args.excluded_output.resolve()
        if args.excluded_output
        else derived_sidecar(output, "_excluded_records.csv")
    )
    report_output = (
        args.report_output.resolve()
        if args.report_output
        else derived_sidecar(output, "_generation_report.json")
    )
    partial = partial_path_for(output)

    if output.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("--output must end in .h5 or .hdf5")
    if path_is_within(output, resized_dir) or path_is_within(partial, resized_dir):
        raise ValueError("HDF5 output must not be inside the resized image directory")

    protected_inputs = {metadata, resize_manifest, resized_dir}
    for candidate in (output, partial, excluded_output, report_output):
        if candidate in protected_inputs:
            raise ValueError(f"Output conflicts with an input path: {candidate}")

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    return (
        metadata,
        resize_manifest,
        resized_dir,
        output,
        partial,
        excluded_output,
        report_output,
    )


def load_resize_manifest(path: Path) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sampleid",
            "source_image_file",
            "resized_image_file",
            "encoded_bytes",
            "status",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Resize manifest is missing columns: {sorted(missing)}")
        for row in reader:
            sampleid = row["sampleid"].strip()
            if not sampleid:
                raise ValueError("Resize manifest contains a blank sampleid")
            if sampleid in manifest:
                raise ValueError(f"Duplicate sampleid in resize manifest: {sampleid}")
            if row["status"] not in {"created", "reused"}:
                raise ValueError(
                    f"Resize manifest has unsuccessful row for {sampleid}: "
                    f"{row['status']}"
                )
            manifest[sampleid] = row
    if not manifest:
        raise ValueError(f"Resize manifest contains no records: {path}")
    return manifest


def exclusion_row(
    row: dict[str, str], fieldnames: list[str], reasons: list[str]
) -> dict[str, str]:
    result = {field: row.get(field, "") for field in fieldnames}
    result["exclusion_reason"] = ";".join(reasons)
    return result


def load_records(
    metadata: Path,
    resized_dir: Path,
    manifest: dict[str, dict[str, str]],
    limit: int | None,
) -> tuple[
    list[Record],
    list[dict[str, str]],
    list[str],
    dict[str, int],
    int,
    str,
]:
    records: list[Record] = []
    excluded: list[dict[str, str]] = []
    missing_taxonomy = {field: 0 for field in TAXONOMY_FIELDS}
    missing_taxonomy["dna_bin"] = 0
    seen_sampleids: set[str] = set()
    seen_processids: set[str] = set()
    eligible_total = 0
    fingerprint = hashlib.sha256()

    with metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
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
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

        for csv_row_number, row in enumerate(reader, 2):
            if row["marker_code"].strip() != MARKER_CODE:
                continue

            sampleid = row["sampleid"].strip()
            processid = row["processid"].strip()
            source_image_file = row["image_file"].strip()
            barcode = row["dna_barcode"].strip()
            reasons = []
            if not sampleid:
                reasons.append("missing_sampleid")
            if not processid:
                reasons.append("missing_processid")
            if not source_image_file:
                reasons.append("missing_image_file")
            if not barcode:
                reasons.append("missing_dna_barcode")

            if sampleid:
                if sampleid in seen_sampleids:
                    raise ValueError(
                        f"Duplicate COI-5P sampleid at CSV row "
                        f"{csv_row_number}: {sampleid}"
                    )
                seen_sampleids.add(sampleid)
            if processid:
                if processid in seen_processids:
                    raise ValueError(
                        f"Duplicate COI-5P processid at CSV row "
                        f"{csv_row_number}: {processid}"
                    )
                seen_processids.add(processid)

            if reasons:
                excluded.append(exclusion_row(row, fieldnames, reasons))
                continue

            resize_row = manifest.get(sampleid)
            if resize_row is None:
                raise ValueError(f"No resize manifest entry for {sampleid}")
            if resize_row["source_image_file"] != source_image_file:
                raise ValueError(
                    f"Source filename mismatch for {sampleid}: metadata has "
                    f"{source_image_file!r}, manifest has "
                    f"{resize_row['source_image_file']!r}"
                )
            expected_resized_name = f"{Path(source_image_file).stem}.png"
            resized_name = resize_row["resized_image_file"]
            if resized_name != expected_resized_name:
                raise ValueError(
                    f"Resized filename mismatch for {sampleid}: "
                    f"{resized_name!r} != {expected_resized_name!r}"
                )

            resized_path = resized_dir / resized_name
            if not resized_path.is_file():
                raise FileNotFoundError(f"Missing resized image: {resized_path}")
            encoded_bytes = int(resize_row["encoded_bytes"])
            if resized_path.stat().st_size != encoded_bytes:
                raise ValueError(
                    f"Image size changed after resize for {sampleid}: "
                    f"manifest={encoded_bytes}, current={resized_path.stat().st_size}"
                )

            eligible_total += 1
            if limit is not None and len(records) >= limit:
                continue

            values = {
                field: normalized_taxonomy(row[field])
                for field in TAXONOMY_FIELDS
            }
            for field in TAXONOMY_FIELDS:
                if not row[field].strip():
                    missing_taxonomy[field] += 1
            dna_bin = row["dna_bin"].strip()
            if not dna_bin:
                missing_taxonomy["dna_bin"] += 1
                dna_bin = NOT_CLASSIFIED

            record = Record(
                sampleid=sampleid,
                processid=processid,
                barcode=barcode,
                dna_bin=dna_bin,
                order=values["order"],
                family=values["family"],
                genus=values["genus"],
                species=values["species"],
                image_file=resized_name,
                source_image_file=source_image_file,
                record_id=row["record_id"].strip(),
                specimenid=row["specimenid"].strip(),
                marker_code=MARKER_CODE,
                encoded_bytes=encoded_bytes,
            )
            records.append(record)
            # Cover every value persisted to HDF5 so a resumed run cannot
            # silently combine a partial file with changed metadata.
            for field in STRING_FIELDS:
                value = getattr(record, field)
                fingerprint.update(value.encode("utf-8"))
                fingerprint.update(b"\0")
            fingerprint.update(str(record.encoded_bytes).encode("ascii"))
            fingerprint.update(b"\0")

    if not records:
        raise ValueError("No eligible records selected")
    return (
        records,
        excluded,
        fieldnames,
        missing_taxonomy,
        eligible_total,
        fingerprint.hexdigest(),
    )


def string_values(records: list[Record], field: str) -> np.ndarray:
    return np.asarray([getattr(record, field) for record in records], dtype=object)


def create_partial_hdf5(
    partial: Path,
    records: list[Record],
    fingerprint: str,
    metadata: Path,
    resize_manifest: Path,
    resized_dir: Path,
    compression: str,
    limit: int | None,
) -> h5py.File:
    count = len(records)
    maximum_encoded_bytes = max(record.encoded_bytes for record in records)
    file = h5py.File(partial, "w", libver="latest")
    try:
        file.attrs.update(
            {
                "format_name": "CLIBD CBG COI-5P",
                "format_version": 1,
                "complete": False,
                "selection_fingerprint_sha256": fingerprint,
                "metadata_path": str(metadata),
                "resize_manifest_path": str(resize_manifest),
                "resized_image_dir": str(resized_dir),
                "marker_filter": MARKER_CODE,
                "group_name": GROUP_NAME,
                "selected_records": count,
                "limit": -1 if limit is None else limit,
                "created_unix_time": time.time(),
            }
        )
        group = file.create_group(GROUP_NAME)
        group.attrs["rows_written"] = 0
        group.attrs["maximum_encoded_bytes"] = maximum_encoded_bytes
        image_options: dict[str, Any] = {
            "shape": (count, maximum_encoded_bytes),
            "dtype": np.uint8,
            "chunks": (1, maximum_encoded_bytes),
        }
        if compression == "lzf":
            image_options["compression"] = "lzf"
        group.create_dataset("image", **image_options)
        group.create_dataset(
            "image_mask",
            data=np.asarray(
                [record.encoded_bytes for record in records], dtype=np.int64
            ),
            dtype=np.int64,
        )
        text_dtype = h5py.string_dtype(encoding="utf-8")
        for field in STRING_FIELDS:
            group.create_dataset(
                field,
                data=string_values(records, field),
                dtype=text_dtype,
            )
        file.flush()
        return file
    except Exception:
        file.close()
        raise


def open_resumable_partial(
    partial: Path,
    records: list[Record],
    fingerprint: str,
) -> h5py.File:
    file = h5py.File(partial, "r+", libver="latest")
    try:
        if file.attrs.get("selection_fingerprint_sha256", "") != fingerprint:
            raise ValueError(
                "Partial HDF5 selection does not match current inputs; use "
                "--restart-partial only after inspecting the partial file"
            )
        group = file[GROUP_NAME]
        expected_shape = (
            len(records),
            max(record.encoded_bytes for record in records),
        )
        if group["image"].shape != expected_shape:
            raise ValueError(
                f"Partial image shape {group['image'].shape} != {expected_shape}"
            )
        rows_written = int(group.attrs["rows_written"])
        if not 0 <= rows_written <= len(records):
            raise ValueError(f"Invalid rows_written in partial file: {rows_written}")
        return file
    except Exception:
        file.close()
        raise


def read_png(task: tuple[Path, int]) -> bytes:
    path, expected_bytes = task
    with path.open("rb") as handle:
        payload = handle.read()
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Image byte length changed for {path}: "
            f"{len(payload)} != {expected_bytes}"
        )
    if not payload.startswith(PNG_SIGNATURE):
        raise ValueError(f"Image does not have a PNG signature: {path}")
    return payload


def write_images(
    file: h5py.File,
    records: list[Record],
    resized_dir: Path,
    batch_size: int,
    read_workers: int,
) -> tuple[int, float]:
    group = file[GROUP_NAME]
    dataset = group["image"]
    start = int(group.attrs["rows_written"])
    started = time.monotonic()
    if start:
        print(f"resume_from_row={start}", flush=True)

    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        for batch_start in range(start, len(records), batch_size):
            batch_end = min(batch_start + batch_size, len(records))
            batch_records = records[batch_start:batch_end]
            tasks = [
                (resized_dir / record.image_file, record.encoded_bytes)
                for record in batch_records
            ]
            payloads = list(executor.map(read_png, tasks))
            padded = np.zeros(
                (len(batch_records), dataset.shape[1]), dtype=np.uint8
            )
            for index, payload in enumerate(payloads):
                padded[index, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)
            dataset[batch_start:batch_end, :] = padded
            group.attrs.modify("rows_written", batch_end)
            file.flush()

            elapsed = time.monotonic() - started
            processed_this_run = batch_end - start
            rate = processed_this_run / elapsed if elapsed else 0.0
            remaining = (len(records) - batch_end) / rate if rate else float("inf")
            print(
                f"written={batch_end}/{len(records)} "
                f"rate={rate:.2f}/s eta_minutes={remaining / 60:.1f}",
                flush=True,
            )
    return start, time.monotonic() - started


def decode_hdf5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def validation_indices(count: int, requested: int) -> list[int]:
    if requested >= count:
        return list(range(count))
    if requested == 1:
        return [0]
    return sorted(
        set(np.linspace(0, count - 1, num=requested, dtype=np.int64).tolist())
    )


def validate_hdf5(
    file: h5py.File,
    records: list[Record],
    resized_dir: Path,
    requested_samples: int,
) -> int:
    group = file[GROUP_NAME]
    expected = set(STRING_FIELDS) | {"image", "image_mask"}
    if set(group.keys()) != expected:
        raise ValueError(
            f"Unexpected HDF5 datasets: {sorted(set(group.keys()) ^ expected)}"
        )
    for name in expected:
        if len(group[name]) != len(records):
            raise ValueError(
                f"Dataset {name} has {len(group[name])} rows; "
                f"expected {len(records)}"
            )
    if int(group.attrs["rows_written"]) != len(records):
        raise ValueError("Not all image rows were written")

    indices = validation_indices(len(records), requested_samples)
    for index in indices:
        length = int(group["image_mask"][index])
        if length != records[index].encoded_bytes:
            raise ValueError(f"image_mask mismatch at row {index}")
        encoded_padded = group["image"][index].astype(np.uint8)
        encoded = encoded_padded[:length].tobytes()
        source_payload = (resized_dir / records[index].image_file).read_bytes()
        if encoded != source_payload:
            raise ValueError(f"HDF5 image bytes differ at row {index}")
        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError(f"Non-PNG image at row {index}: {image.format}")
            if image.mode != "RGB":
                raise ValueError(f"Non-RGB image at row {index}: {image.mode}")
            if min(image.size) != 256:
                raise ValueError(
                    f"Image short edge is not 256 at row {index}: {image.size}"
                )
        if not decode_hdf5_string(group["barcode"][index]):
            raise ValueError(f"Blank barcode at row {index}")
        for field in TAXONOMY_FIELDS:
            if not decode_hdf5_string(group[field][index]):
                raise ValueError(f"Blank {field} at row {index}")
    return len(indices)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.read_workers <= 0:
        raise ValueError("--read-workers must be positive")
    if args.validation_samples <= 0:
        raise ValueError("--validation-samples must be positive")

    (
        metadata,
        resize_manifest,
        resized_dir,
        output,
        partial,
        excluded_output,
        report_output,
    ) = validate_paths(args)
    if args.restart_partial and partial.exists():
        partial.unlink()

    print(f"metadata={metadata}", flush=True)
    print(f"resize_manifest={resize_manifest}", flush=True)
    print(f"resized_dir={resized_dir} (read-only)", flush=True)
    print(f"output={output}", flush=True)
    print(f"partial={partial}", flush=True)

    manifest = load_resize_manifest(resize_manifest)
    (
        records,
        excluded,
        metadata_fieldnames,
        missing_taxonomy,
        eligible_total,
        fingerprint,
    ) = load_records(metadata, resized_dir, manifest, args.limit)
    maximum_encoded_bytes = max(record.encoded_bytes for record in records)
    total_encoded_bytes = sum(record.encoded_bytes for record in records)
    print(f"eligible_total={eligible_total}", flush=True)
    print(f"selected_records={len(records)}", flush=True)
    print(f"excluded_records={len(excluded)}", flush=True)
    print(f"maximum_encoded_bytes={maximum_encoded_bytes}", flush=True)

    start_time = time.monotonic()
    if partial.exists():
        file = open_resumable_partial(partial, records, fingerprint)
    else:
        file = create_partial_hdf5(
            partial,
            records,
            fingerprint,
            metadata,
            resize_manifest,
            resized_dir,
            args.compression,
            args.limit,
        )

    try:
        resumed_from, write_elapsed = write_images(
            file,
            records,
            resized_dir,
            args.batch_size,
            args.read_workers,
        )
        validated_samples = validate_hdf5(
            file, records, resized_dir, args.validation_samples
        )
        file.attrs.modify("complete", True)
        file.attrs["validated_samples"] = validated_samples
        file.attrs["completed_unix_time"] = time.time()
        file.flush()
    finally:
        file.close()

    elapsed = time.monotonic() - start_time
    report = {
        "metadata": str(metadata),
        "resize_manifest": str(resize_manifest),
        "resized_image_dir": str(resized_dir),
        "output": str(output),
        "group": GROUP_NAME,
        "marker_filter": MARKER_CODE,
        "coi5p_manifest_records": len(manifest),
        "eligible_records": eligible_total,
        "selected_records": len(records),
        "excluded_records": len(excluded),
        "limit": args.limit,
        "missing_values_replaced_with_not_classified": missing_taxonomy,
        "datasets": list(STRING_FIELDS) + ["image", "image_mask"],
        "language_token_datasets_stored": False,
        "image_format": "PNG",
        "image_mode": "RGB",
        "image_short_edge": 256,
        "images_reencoded_during_hdf5_generation": False,
        "maximum_encoded_bytes": maximum_encoded_bytes,
        "total_encoded_bytes": total_encoded_bytes,
        "image_dataset_compression": args.compression,
        "resumed_from_row": resumed_from,
        "write_elapsed_seconds": write_elapsed,
        "total_elapsed_seconds": elapsed,
        "validated_samples": validated_samples,
        "selection_fingerprint_sha256": fingerprint,
        "hdf5_file_bytes": partial.stat().st_size,
    }
    atomic_write_csv(
        excluded_output,
        metadata_fieldnames + ["exclusion_reason"],
        excluded,
    )
    atomic_write_json(report_output, report)
    os.replace(partial, output)
    print(f"excluded_output={excluded_output}", flush=True)
    print(f"report_output={report_output}", flush=True)
    print(f"hdf5_output={output}", flush=True)
    print("hdf5_status=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
