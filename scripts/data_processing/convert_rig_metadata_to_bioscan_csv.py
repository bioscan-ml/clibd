#!/usr/bin/env python3
"""Convert tab-separated RIG metadata to a BIOSCAN-5M-compatible CSV.

The input contains one row per sequence record, so a specimen can appear more
than once when it has multiple genetic markers. This converter preserves every
sequence record. The first 21 output columns match BIOSCAN-5M v3.4, followed by
record_id, marker_code, nuc_basecount, and specimenid so different marker rows
for the same image remain distinguishable. Images are matched by the exact,
case-sensitive filename stem == sampleid rule.

The implementation uses two streaming passes over the metadata rather than
loading the 600+ MB source table into a pandas DataFrame.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterable, TextIO


REFERENCE_COLUMNS = [
    "processid",
    "sampleid",
    "image_file",
    "chunk_number",
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "genus",
    "species",
    "dna_bin",
    "dna_barcode",
    "split",
    "country",
    "province_state",
    "coord-lat",
    "coord-lon",
    "surface_area",
    "bioscan1M_index",
    "label_was_inferred",
]

AUDIT_COLUMNS = ["record_id", "marker_code", "nuc_basecount", "specimenid"]
OUTPUT_COLUMNS = REFERENCE_COLUMNS + AUDIT_COLUMNS

REQUIRED_INPUT_COLUMNS = {
    "processid",
    "record_id",
    "sampleid",
    "specimenid",
    "marker_code",
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "genus",
    "species",
    "bin_uri",
    "nuc",
    "nuc_basecount",
    "coord",
    "country.ocean",
    "province.state",
}

QUALITY_FIELDS = [
    "phylum",
    "class",
    "order",
    "family",
    "subfamily",
    "genus",
    "species",
    "bin_uri",
    "nuc",
    "coord",
    "country.ocean",
    "province.state",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg"}


class ConversionError(RuntimeError):
    """Raised when an input invariant needed for a safe conversion fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RIG TSV metadata to the BIOSCAN-5M v3.4 CSV schema."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    image_source = parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument(
        "--archive-file-list",
        type=Path,
        help="unzip -Z1 style archive member list",
    )
    image_source.add_argument(
        "--image-dir",
        type=Path,
        help="scan an extracted image directory instead of the archive member list",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        help=(
            "optional BIOSCAN-5M CSV whose header is checked against the "
            "built-in 21-column schema"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--unmatched-output",
        type=Path,
        help="default: <output stem>_unmatched_images.csv",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        help="default: <output stem>_quality_report.json",
    )
    parser.add_argument("--split", default="no_split")
    parser.add_argument("--chunk-number", default="1")
    parser.add_argument(
        "--label-was-inferred",
        default="",
        help="leave empty unless the label provenance is known (for example, 0 or 1)",
    )
    parser.add_argument(
        "--fail-on-unmatched-images",
        action="store_true",
        help="fail instead of reporting image filenames with no exact sampleid match",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files after a successful conversion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and validate everything without creating output files",
    )
    return parser.parse_args()


def derived_output_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def validate_reference_header(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if header != REFERENCE_COLUMNS:
        raise ConversionError(
            f"Reference header does not match the expected 21-column prefix: {path}\n"
            f"expected={REFERENCE_COLUMNS!r}\nactual={header!r}"
        )


def iter_image_names_from_archive_list(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            member = raw_line.rstrip("\r\n")
            if not member or member.endswith("/"):
                continue
            name = PurePosixPath(member).name
            if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES:
                yield name


def iter_image_names_from_directory(path: Path) -> Iterable[str]:
    for candidate in path.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            yield candidate.name


def load_image_inventory(args: argparse.Namespace) -> tuple[dict[str, str], int]:
    if args.image_dir is not None:
        source = args.image_dir.resolve(strict=True)
        if not source.is_dir():
            raise ConversionError(f"Image path is not a directory: {source}")
        names = iter_image_names_from_directory(source)
    else:
        source = args.archive_file_list.resolve(strict=True)
        if not source.is_file():
            raise ConversionError(f"Archive member list is not a file: {source}")
        names = iter_image_names_from_archive_list(source)

    by_stem: dict[str, str] = {}
    count = 0
    for name in names:
        count += 1
        stem = PurePosixPath(name).stem
        previous = by_stem.get(stem)
        if previous is not None:
            raise ConversionError(
                f"Multiple image files share the same stem {stem!r}: "
                f"{previous!r}, {name!r}"
            )
        by_stem[stem] = name
    if not by_stem:
        raise ConversionError(f"No JPEG images found in {source}")
    return by_stem, count


def read_header(reader: csv.reader, metadata: Path) -> tuple[list[str], dict[str, int]]:
    header = next(reader, None)
    if header is None:
        raise ConversionError(f"Metadata is empty: {metadata}")
    if len(header) != len(set(header)):
        raise ConversionError("Metadata header contains duplicate column names")
    missing = sorted(REQUIRED_INPUT_COLUMNS - set(header))
    if missing:
        raise ConversionError(f"Metadata is missing required columns: {missing}")
    return header, {name: index for index, name in enumerate(header)}


def inspect_metadata_rows(metadata: Path) -> tuple[set[tuple[str, str]], set[str], dict]:
    record_keys: set[tuple[str, str]] = set()
    sample_counts: collections.Counter[str] = collections.Counter()
    marker_counts: collections.Counter[str] = collections.Counter()
    row_width_counts: collections.Counter[int] = collections.Counter()
    total_rows = 0

    with metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header, index = read_header(reader, metadata)
        expected_width = len(header)
        for line_number, row in enumerate(reader, 2):
            total_rows += 1
            row_width_counts[len(row)] += 1
            if len(row) != expected_width:
                raise ConversionError(
                    f"Malformed metadata row at line {line_number}: "
                    f"expected {expected_width} columns, got {len(row)}"
                )
            sampleid = row[index["sampleid"]].strip()
            if not sampleid:
                raise ConversionError(f"Blank sampleid at line {line_number}")
            marker = row[index["marker_code"]].strip()
            marker_counts[marker or "<blank>"] += 1
            sample_counts[sampleid] += 1
            record_key = (sampleid, marker)
            if record_key in record_keys:
                raise ConversionError(
                    "Duplicate (sampleid, marker_code) key at line "
                    f"{line_number}: {record_key!r}"
                )
            record_keys.add(record_key)

    duplicate_groups = sum(1 for count in sample_counts.values() if count > 1)
    duplicate_extra_rows = sum(count - 1 for count in sample_counts.values())
    return record_keys, set(sample_counts), {
        "metadata_rows": total_rows,
        "metadata_columns": len(header),
        "row_width_counts": dict(sorted(row_width_counts.items())),
        "raw_marker_counts": dict(marker_counts.most_common()),
        "unique_sampleids": len(sample_counts),
        "unique_sampleid_marker_keys": len(record_keys),
        "duplicate_sampleid_groups": duplicate_groups,
        "duplicate_extra_rows": duplicate_extra_rows,
        "maximum_records_per_sampleid": max(sample_counts.values(), default=0),
    }


def split_coord(value: str, *, sampleid: str) -> tuple[str, str]:
    value = value.strip()
    if not value:
        return "", ""
    pieces = [piece.strip() for piece in value.split(",")]
    if len(pieces) != 2:
        raise ConversionError(f"Invalid coord for {sampleid}: {value!r}")
    try:
        latitude, longitude = map(float, pieces)
    except ValueError as exc:
        raise ConversionError(f"Non-numeric coord for {sampleid}: {value!r}") from exc
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ConversionError(f"Out-of-range coord for {sampleid}: {value!r}")
    return pieces[0], pieces[1]


def atomic_text_file(path: Path) -> tuple[TextIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    return os.fdopen(descriptor, "w", encoding="utf-8", newline=""), Path(temporary_name)


def output_row(
    row: list[str],
    index: dict[str, int],
    image_file: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    sampleid = row[index["sampleid"]].strip()
    latitude, longitude = split_coord(row[index["coord"]], sampleid=sampleid)
    return {
        "processid": row[index["processid"]].strip(),
        "sampleid": sampleid,
        "image_file": image_file,
        "chunk_number": args.chunk_number,
        "phylum": row[index["phylum"]].strip(),
        "class": row[index["class"]].strip(),
        "order": row[index["order"]].strip(),
        "family": row[index["family"]].strip(),
        "subfamily": row[index["subfamily"]].strip(),
        "genus": row[index["genus"]].strip(),
        "species": row[index["species"]].strip(),
        "dna_bin": row[index["bin_uri"]].strip(),
        "dna_barcode": "".join(row[index["nuc"]].split()).upper(),
        "split": args.split,
        "country": row[index["country.ocean"]].strip(),
        "province_state": row[index["province.state"]].strip(),
        "coord-lat": latitude,
        "coord-lon": longitude,
        "surface_area": "",
        "bioscan1M_index": "",
        "label_was_inferred": args.label_was_inferred,
        "record_id": row[index["record_id"]].strip(),
        "marker_code": row[index["marker_code"]].strip(),
        "nuc_basecount": row[index["nuc_basecount"]].strip(),
        "specimenid": row[index["specimenid"]].strip(),
    }


def convert_metadata_rows(
    args: argparse.Namespace,
    metadata: Path,
    images_by_stem: dict[str, str],
    writer: csv.DictWriter | None,
) -> dict:
    missing_counts: collections.Counter[str] = collections.Counter()
    selected_marker_counts: collections.Counter[str] = collections.Counter()
    metadata_without_image: list[dict[str, str]] = []
    barcode_length_mismatch_count = 0
    barcode_length_mismatch_samples: list[dict[str, object]] = []
    barcode_iupac_ambiguity_count = 0
    barcode_iupac_ambiguity_samples: list[dict[str, object]] = []
    input_rows = output_rows = 0

    with metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header, index = read_header(reader, metadata)
        for row in reader:
            sampleid = row[index["sampleid"]].strip()
            input_rows += 1
            marker = row[index["marker_code"]].strip() or "<blank>"
            selected_marker_counts[marker] += 1
            for field in QUALITY_FIELDS:
                if not row[index[field]].strip():
                    missing_counts[field] += 1

            barcode = "".join(row[index["nuc"]].split()).upper()
            declared_count = row[index["nuc_basecount"]].strip()
            if barcode and declared_count:
                try:
                    length_differs = len(barcode) != int(declared_count)
                except ValueError:
                    length_differs = True
                if length_differs:
                    barcode_length_mismatch_count += 1
                    if len(barcode_length_mismatch_samples) < 20:
                        barcode_length_mismatch_samples.append(
                            {
                                "sampleid": sampleid,
                                "actual_length": len(barcode),
                                "nuc_basecount": declared_count,
                            }
                        )
            ambiguity_codes = sorted(set(barcode) - set("ACGTN-"))
            if ambiguity_codes:
                barcode_iupac_ambiguity_count += 1
                if len(barcode_iupac_ambiguity_samples) < 20:
                    barcode_iupac_ambiguity_samples.append(
                        {"sampleid": sampleid, "codes": ambiguity_codes}
                    )

            image_file = images_by_stem.get(sampleid)
            if image_file is None:
                if len(metadata_without_image) < 100:
                    metadata_without_image.append(
                        {
                            "sampleid": sampleid,
                            "processid": row[index["processid"]].strip(),
                        }
                    )
                continue
            if writer is not None:
                writer.writerow(output_row(row, index, image_file, args))
            output_rows += 1

    return {
        "input_rows_reprocessed": input_rows,
        "output_rows": output_rows,
        "output_marker_counts": dict(selected_marker_counts.most_common()),
        "output_source_missing_counts": dict(sorted(missing_counts.items())),
        "metadata_rows_without_exact_image_count": input_rows - output_rows,
        "metadata_without_exact_image_samples": metadata_without_image,
        "barcode_length_mismatch_count": barcode_length_mismatch_count,
        "barcode_length_mismatch_samples": barcode_length_mismatch_samples,
        "barcode_iupac_ambiguity_count": barcode_iupac_ambiguity_count,
        "barcode_iupac_ambiguity_samples": barcode_iupac_ambiguity_samples,
    }


def ensure_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise ConversionError(
            "Refusing to overwrite existing output files. Use --overwrite after review:\n"
            f"{formatted}"
        )


def main() -> None:
    args = parse_args()
    metadata = args.metadata.resolve(strict=True)
    reference = (
        args.reference_csv.resolve(strict=True) if args.reference_csv else None
    )
    output = args.output.expanduser().resolve()
    unmatched_output = (
        args.unmatched_output.expanduser().resolve()
        if args.unmatched_output
        else derived_output_path(output, "_unmatched_images.csv")
    )
    quality_report = (
        args.quality_report.expanduser().resolve()
        if args.quality_report
        else derived_output_path(output, "_quality_report.json")
    )

    print(f"metadata={metadata}")
    print(f"reference_csv={reference}")
    print(f"output={output}")
    print(f"unmatched_output={unmatched_output}")
    print(f"quality_report={quality_report}")
    print(f"dry_run={args.dry_run}")

    if not metadata.is_file():
        raise ConversionError(f"Metadata is not a file: {metadata}")
    if reference is not None:
        validate_reference_header(reference)
    images_by_stem, image_count = load_image_inventory(args)
    record_keys, sampleids, report = inspect_metadata_rows(metadata)

    unmatched_image_stems = sorted(set(images_by_stem) - sampleids)
    unmatched_images = [images_by_stem[stem] for stem in unmatched_image_stems]
    if unmatched_images and args.fail_on_unmatched_images:
        raise ConversionError(
            f"Found {len(unmatched_images)} images with no exact sampleid match"
        )

    output_paths = [output, unmatched_output, quality_report]
    if not args.dry_run:
        ensure_outputs_available(output_paths, args.overwrite)

    main_handle = None
    main_temp = None
    unmatched_handle = None
    unmatched_temp = None
    report_temp = None
    try:
        writer = None
        if not args.dry_run:
            main_handle, main_temp = atomic_text_file(output)
            writer = csv.DictWriter(
                main_handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()

        report.update(
            convert_metadata_rows(args, metadata, images_by_stem, writer)
        )
        report.update(
            {
                "metadata_path": str(metadata),
                "reference_csv": str(reference) if reference is not None else None,
                "output_schema": OUTPUT_COLUMNS,
                "image_count": image_count,
                "unique_image_stems": len(images_by_stem),
                "unmatched_image_count": len(unmatched_images),
                "unmatched_image_samples": unmatched_images[:100],
                "split_value": args.split,
                "chunk_number_value": args.chunk_number,
                "label_was_inferred_value": args.label_was_inferred,
                "surface_area_policy": "blank_not_available_in_source_metadata",
                "bioscan1M_index_policy": "blank_not_applicable_to_new_dataset",
                "record_policy": "preserve_all_sequence_records",
                "record_key": ["sampleid", "marker_code"],
                "record_key_count": len(record_keys),
                "image_match_policy": "exact case-sensitive image stem equals sampleid",
            }
        )

        if not args.dry_run:
            assert main_handle is not None and main_temp is not None
            main_handle.flush()
            os.fsync(main_handle.fileno())
            main_handle.close()
            main_handle = None

            unmatched_handle, unmatched_temp = atomic_text_file(unmatched_output)
            unmatched_writer = csv.DictWriter(
                unmatched_handle,
                fieldnames=["image_file", "image_stem", "issue"],
                lineterminator="\n",
            )
            unmatched_writer.writeheader()
            for stem in unmatched_image_stems:
                unmatched_writer.writerow(
                    {
                        "image_file": images_by_stem[stem],
                        "image_stem": stem,
                        "issue": "no_exact_sampleid_match",
                    }
                )
            unmatched_handle.flush()
            os.fsync(unmatched_handle.fileno())
            unmatched_handle.close()
            unmatched_handle = None

            report_handle, report_temp = atomic_text_file(quality_report)
            json.dump(report, report_handle, indent=2, ensure_ascii=False, sort_keys=True)
            report_handle.write("\n")
            report_handle.flush()
            os.fsync(report_handle.fileno())
            report_handle.close()

            os.replace(main_temp, output)
            main_temp = None
            os.replace(unmatched_temp, unmatched_output)
            unmatched_temp = None
            os.replace(report_temp, quality_report)
            report_temp = None

        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    finally:
        for handle in (main_handle, unmatched_handle):
            if handle is not None:
                handle.close()
        for temporary in (main_temp, unmatched_temp, report_temp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (ConversionError, OSError, csv.Error) as exc:
        raise SystemExit(f"ERROR: {exc}")
