#!/usr/bin/env python3
"""Resize CBG COI-5P images losslessly for CLIBD HDF5 generation.

The output images have a 256-pixel short edge, preserve their aspect ratio,
and are stored as RGB PNG files to avoid an additional lossy JPEG encoding.
Individual files are written atomically, and valid existing outputs are reused
so interrupted runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image, ImageOps


MARKER = "COI-5P"

MANIFEST_FIELDS = [
    "row_index",
    "sampleid",
    "source_image_file",
    "resized_image_file",
    "source_width",
    "source_height",
    "resized_width",
    "resized_height",
    "encoded_bytes",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize COI-5P images to a 256-pixel short edge and save lossless PNGs."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-edge", type=int, default=256)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--compression-level", type=int, default=6, choices=range(10))
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument(
        "--limit",
        type=int,
        help="process only the first N eligible records (for smoke testing)",
    )
    parser.add_argument(
        "--overwrite-completed-run",
        action="store_true",
        help="allow rebuilding reports when a completed manifest already exists",
    )
    return parser.parse_args()


def read_coi5p_records(metadata: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sampleids: set[str] = set()
    output_names: set[str] = set()

    with metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Metadata has no header: {metadata}")
        required = {"sampleid", "image_file", "marker_code"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Metadata is missing required columns: {missing}")

        for csv_row_number, row in enumerate(reader, 2):
            if row["marker_code"].strip() != MARKER:
                continue
            sampleid = row["sampleid"].strip()
            source_image_file = row["image_file"].strip()
            if not sampleid:
                raise ValueError(f"Blank sampleid at CSV row {csv_row_number}")
            if not source_image_file:
                raise ValueError(f"Blank image_file at CSV row {csv_row_number}")
            if sampleid in sampleids:
                raise ValueError(f"Duplicate COI-5P sampleid: {sampleid}")

            resized_image_file = f"{Path(source_image_file).stem}.png"
            if resized_image_file in output_names:
                raise ValueError(
                    f"Multiple source files map to {resized_image_file!r}"
                )

            sampleids.add(sampleid)
            output_names.add(resized_image_file)
            records.append(
                {
                    "row_index": len(records),
                    "sampleid": sampleid,
                    "source_image_file": source_image_file,
                    "resized_image_file": resized_image_file,
                }
            )
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise ValueError(f"No {MARKER} records found in {metadata}")
    return records


def inspect_png(path: Path, short_edge: int) -> tuple[int, int, int]:
    with Image.open(path) as image:
        image_format = image.format
        mode = image.mode
        width, height = image.size
        image.verify()
    if image_format != "PNG":
        raise ValueError(f"Expected PNG, got {image_format}")
    if mode != "RGB":
        raise ValueError(f"Expected RGB, got {mode}")
    if min(width, height) != short_edge:
        raise ValueError(
            f"Expected short edge {short_edge}, got dimensions {width}x{height}"
        )
    return width, height, path.stat().st_size


def resize_one(task: tuple[dict[str, Any], str, str, str, int, int]) -> dict[str, Any]:
    (
        record,
        source_dir_string,
        output_dir_string,
        temporary_dir_string,
        short_edge,
        compression_level,
    ) = task
    source_dir = Path(source_dir_string)
    output_dir = Path(output_dir_string)
    temporary_dir = Path(temporary_dir_string)
    source_path = source_dir / record["source_image_file"]
    output_path = output_dir / record["resized_image_file"]
    temporary_path = temporary_dir / (
        f"{record['resized_image_file']}.{os.getpid()}.partial"
    )

    result = dict(record)
    try:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        # This trusted scientific dataset contains a small number of legitimate
        # 180M-344M-pixel RIG photographs. ProcessPool workers are isolated, so
        # disabling Pillow's decompression-bomb threshold here affects only the
        # explicitly requested local source image.
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(source_path) as source:
            oriented = ImageOps.exif_transpose(source)
            source_width, source_height = oriented.size

            if output_path.exists():
                resized_width, resized_height, encoded_bytes = inspect_png(
                    output_path, short_edge
                )
                status = "reused"
            else:
                image = oriented.convert("RGB")
                scale = short_edge / min(image.width, image.height)
                resized_width = max(1, round(image.width * scale))
                resized_height = max(1, round(image.height * scale))
                image = image.resize(
                    (resized_width, resized_height),
                    Image.Resampling.BILINEAR,
                )
                image.save(
                    temporary_path,
                    format="PNG",
                    compress_level=compression_level,
                )
                (
                    verified_width,
                    verified_height,
                    encoded_bytes,
                ) = inspect_png(temporary_path, short_edge)
                if (verified_width, verified_height) != (
                    resized_width,
                    resized_height,
                ):
                    raise ValueError("Saved PNG dimensions changed unexpectedly")
                os.replace(temporary_path, output_path)
                status = "created"

        result.update(
            {
                "source_width": source_width,
                "source_height": source_height,
                "resized_width": resized_width,
                "resized_height": resized_height,
                "encoded_bytes": encoded_bytes,
                "status": status,
                "error": "",
            }
        )
    except Exception as error:  # keep the batch running and report every failure
        temporary_path.unlink(missing_ok=True)
        result.update(
            {
                "source_width": "",
                "source_height": "",
                "resized_width": "",
                "resized_height": "",
                "encoded_bytes": "",
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.partial")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def main() -> None:
    args = parse_args()
    if args.short_edge <= 0:
        raise ValueError("--short-edge must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    metadata = args.metadata.resolve(strict=True)
    source_dir = args.source_dir.resolve(strict=True)
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir / ".partial"
    temporary_dir.mkdir(exist_ok=True)

    manifest = output_dir / "resize_manifest.csv"
    partial_manifest = output_dir / ".resize_manifest.csv.partial"
    failures_path = output_dir / "resize_failures.csv"
    report_path = output_dir / "resize_report.json"
    if manifest.exists() and not args.overwrite_completed_run:
        raise FileExistsError(
            f"Completed manifest already exists: {manifest}. "
            "Use --overwrite-completed-run to revalidate/rebuild reports."
        )

    records = read_coi5p_records(metadata, args.limit)
    tasks = (
        (
            record,
            str(source_dir),
            str(output_dir),
            str(temporary_dir),
            args.short_edge,
            args.compression_level,
        )
        for record in records
    )

    print(f"metadata={metadata}", flush=True)
    print(f"source_dir={source_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"marker={MARKER}", flush=True)
    print(f"records={len(records)}", flush=True)
    print(f"short_edge={args.short_edge}", flush=True)
    print("format=RGB PNG (lossless)", flush=True)
    print(f"workers={args.workers}", flush=True)

    start_time = time.monotonic()
    created = 0
    reused = 0
    failures: list[dict[str, Any]] = []
    total_encoded_bytes = 0
    maximum_encoded_bytes = 0
    maximum_encoded_file = ""

    with partial_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        with Pool(processes=args.workers) as pool:
            for processed, result in enumerate(
                pool.imap(resize_one, tasks, chunksize=1), 1
            ):
                if result["status"] == "failed":
                    failures.append(result)
                else:
                    writer.writerow({field: result[field] for field in MANIFEST_FIELDS})
                    encoded_bytes = int(result["encoded_bytes"])
                    total_encoded_bytes += encoded_bytes
                    if encoded_bytes > maximum_encoded_bytes:
                        maximum_encoded_bytes = encoded_bytes
                        maximum_encoded_file = result["resized_image_file"]
                    if result["status"] == "created":
                        created += 1
                    else:
                        reused += 1

                if processed % args.progress_every == 0 or processed == len(records):
                    handle.flush()
                    elapsed = time.monotonic() - start_time
                    rate = processed / elapsed if elapsed else 0.0
                    remaining = (
                        (len(records) - processed) / rate if rate else float("inf")
                    )
                    print(
                        f"processed={processed}/{len(records)} "
                        f"created={created} reused={reused} "
                        f"failed={len(failures)} rate={rate:.2f}/s "
                        f"eta_minutes={remaining / 60:.1f}",
                        flush=True,
                    )

    failure_fields = MANIFEST_FIELDS + ["error"]
    with failures_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(
            {field: result.get(field, "") for field in failure_fields}
            for result in failures
        )

    elapsed_seconds = time.monotonic() - start_time
    report = {
        "metadata": str(metadata),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "marker_filter": MARKER,
        "requested_records": len(records),
        "successful_records": created + reused,
        "created_records": created,
        "reused_records": reused,
        "failed_records": len(failures),
        "short_edge": args.short_edge,
        "image_mode": "RGB",
        "output_format": "PNG",
        "lossless_output": True,
        "png_compression_level": args.compression_level,
        "total_encoded_bytes": total_encoded_bytes,
        "maximum_encoded_bytes": maximum_encoded_bytes,
        "maximum_encoded_file": maximum_encoded_file,
        "elapsed_seconds": elapsed_seconds,
        "average_images_per_second": len(records) / elapsed_seconds,
        "workers": args.workers,
    }
    atomic_write_json(report_path, report)

    if failures:
        print(
            f"ERROR: {len(failures)} images failed; see {failures_path}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    os.replace(partial_manifest, manifest)
    try:
        temporary_dir.rmdir()
    except OSError:
        pass
    print(f"manifest={manifest}", flush=True)
    print(f"report={report_path}", flush=True)
    print("resize_status=COMPLETE", flush=True)


if __name__ == "__main__":
    main()
