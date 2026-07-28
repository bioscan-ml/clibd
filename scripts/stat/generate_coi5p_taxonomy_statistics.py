#!/usr/bin/env python3
"""Generate COI-5P taxonomy statistics and class-size histograms."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TAXONOMY_COLUMNS = ("order", "family", "genus", "species")
MARKER = "COI-5P"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize taxonomy and missingness for COI-5P records."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "report": output_dir / "coi5p_taxonomy_report.md",
        "json": output_dir / "coi5p_taxonomy_statistics.json",
        "missingness": output_dir / "coi5p_column_missingness.csv",
        "summary": output_dir / "coi5p_taxonomy_unique_summary.csv",
        "overlap": output_dir / "coi5p_bioscan5m_taxonomy_overlap.csv",
        "class_counts": output_dir / "coi5p_taxonomy_class_counts.csv",
        "plot_png": output_dir / "coi5p_taxonomy_class_count_distributions.png",
        "plot_pdf": output_dir / "coi5p_taxonomy_class_count_distributions.pdf",
    }


def validate_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output files already exist; use --overwrite to replace them:\n{joined}"
        )


def scan_csv(
    input_path: Path,
) -> tuple[list[str], int, collections.Counter[str], dict[str, collections.Counter[str]]]:
    missing_counts: collections.Counter[str] = collections.Counter()
    taxon_counts = {column: collections.Counter() for column in TAXONOMY_COLUMNS}
    coi5p_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns:
            raise ValueError(f"Input CSV has no header: {input_path}")
        required = {"marker_code", *TAXONOMY_COLUMNS}
        missing_required = sorted(required - set(columns))
        if missing_required:
            raise ValueError(f"Input CSV is missing columns: {missing_required}")

        for row in reader:
            if row["marker_code"].strip() != MARKER:
                continue
            coi5p_rows += 1
            for column in columns:
                if not row[column].strip():
                    missing_counts[column] += 1
            for column in TAXONOMY_COLUMNS:
                value = row[column].strip()
                if value:
                    taxon_counts[column][value] += 1

    if coi5p_rows == 0:
        raise ValueError(f"No {MARKER} records found in {input_path}")
    return columns, coi5p_rows, missing_counts, taxon_counts


def scan_reference_taxa(reference_path: Path) -> dict[str, set[str]]:
    reference_taxa = {column: set() for column in TAXONOMY_COLUMNS}
    for chunk in pd.read_csv(
        reference_path,
        usecols=list(TAXONOMY_COLUMNS),
        dtype=str,
        keep_default_na=False,
        chunksize=250_000,
    ):
        for column in TAXONOMY_COLUMNS:
            values = chunk[column].str.strip()
            reference_taxa[column].update(values[values != ""].tolist())
    return reference_taxa


def overlap_summary(
    counts: collections.Counter[str], reference_taxa: set[str]
) -> dict[str, float | int]:
    overlapping_taxa = set(counts).intersection(reference_taxa)
    nonmissing_samples = sum(counts.values())
    overlapping_samples = sum(counts[taxon] for taxon in overlapping_taxa)
    return {
        "bioscan5m_unique_taxa": len(reference_taxa),
        "overlapping_taxa": len(overlapping_taxa),
        "new_taxa_overlap_percent": 100 * len(overlapping_taxa) / len(counts),
        "overlapping_samples": overlapping_samples,
        "nonmissing_samples": nonmissing_samples,
        "nonmissing_sample_overlap_percent": 100
        * overlapping_samples
        / nonmissing_samples,
    }


def distribution_summary(counts: collections.Counter[str]) -> dict[str, float | int]:
    sizes = list(counts.values())
    return {
        "unique_taxa": len(sizes),
        "minimum_records_per_taxon": min(sizes),
        "median_records_per_taxon": statistics.median(sizes),
        "mean_records_per_taxon": statistics.mean(sizes),
        "maximum_records_per_taxon": max(sizes),
        "singleton_taxa": sum(size == 1 for size in sizes),
    }


def write_csv_outputs(
    paths: dict[str, Path],
    columns: list[str],
    total: int,
    missing_counts: collections.Counter[str],
    taxon_counts: dict[str, collections.Counter[str]],
    reference_taxa: dict[str, set[str]],
) -> None:
    with paths["missingness"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["column", "missing_count", "missing_percent", "nonmissing_count"],
        )
        writer.writeheader()
        for column in columns:
            missing = missing_counts[column]
            writer.writerow(
                {
                    "column": column,
                    "missing_count": missing,
                    "missing_percent": f"{100 * missing / total:.6f}",
                    "nonmissing_count": total - missing,
                }
            )

    with paths["summary"].open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "rank",
            "unique_taxa",
            "missing_count",
            "missing_percent",
            "minimum_records_per_taxon",
            "median_records_per_taxon",
            "mean_records_per_taxon",
            "maximum_records_per_taxon",
            "singleton_taxa",
            "bioscan5m_unique_taxa",
            "overlapping_taxa",
            "new_taxa_overlap_percent",
            "overlapping_samples",
            "nonmissing_samples",
            "nonmissing_sample_overlap_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank in TAXONOMY_COLUMNS:
            row = distribution_summary(taxon_counts[rank])
            row.update(overlap_summary(taxon_counts[rank], reference_taxa[rank]))
            row.update(
                {
                    "rank": rank,
                    "missing_count": missing_counts[rank],
                    "missing_percent": f"{100 * missing_counts[rank] / total:.6f}",
                    "mean_records_per_taxon": f"{row['mean_records_per_taxon']:.6f}",
                }
            )
            writer.writerow(row)

    with paths["overlap"].open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "rank",
            "new_unique_taxa",
            "bioscan5m_unique_taxa",
            "overlapping_taxa",
            "new_taxa_overlap_percent",
            "new_nonmissing_samples",
            "overlapping_samples",
            "nonmissing_sample_overlap_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank in TAXONOMY_COLUMNS:
            item = overlap_summary(taxon_counts[rank], reference_taxa[rank])
            writer.writerow(
                {
                    "rank": rank,
                    "new_unique_taxa": len(taxon_counts[rank]),
                    "bioscan5m_unique_taxa": item["bioscan5m_unique_taxa"],
                    "overlapping_taxa": item["overlapping_taxa"],
                    "new_taxa_overlap_percent": f"{item['new_taxa_overlap_percent']:.6f}",
                    "new_nonmissing_samples": item["nonmissing_samples"],
                    "overlapping_samples": item["overlapping_samples"],
                    "nonmissing_sample_overlap_percent": (
                        f"{item['nonmissing_sample_overlap_percent']:.6f}"
                    ),
                }
            )

    with paths["class_counts"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "taxon", "record_count"])
        writer.writeheader()
        for rank in TAXONOMY_COLUMNS:
            for taxon, count in sorted(
                taxon_counts[rank].items(), key=lambda item: (-item[1], item[0])
            ):
                writer.writerow({"rank": rank, "taxon": taxon, "record_count": count})


def write_plot(
    paths: dict[str, Path], taxon_counts: dict[str, collections.Counter[str]]
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    for axis, rank in zip(axes.flat, TAXONOMY_COLUMNS):
        bin_counts: collections.Counter[int] = collections.Counter()
        for size in taxon_counts[rank].values():
            bin_counts[size.bit_length() - 1] += 1
        last_bin = max(bin_counts)
        positions = list(range(last_bin + 1))
        heights = [bin_counts[position] for position in positions]
        labels = []
        for position in positions:
            lower = 2**position
            upper = 2 ** (position + 1) - 1
            labels.append(str(lower) if lower == upper else f"{lower}–{upper}")

        axis.bar(
            positions,
            heights,
            width=0.82,
            color="#2878B5",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
        )
        axis.set_yscale("log")
        axis.set_title(f"{rank.capitalize()} (n={len(taxon_counts[rank]):,})")
        axis.set_xlabel("COI-5P records per taxon (power-of-two bins)")
        axis.set_ylabel("Number of taxa (log scale)")
        axis.set_xticks(positions, labels, rotation=45, ha="right", fontsize=8)
        axis.grid(True, axis="y", which="both", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("COI-5P taxonomy class-size histograms", fontsize=15)
    fig.savefig(paths["plot_png"], dpi=220)
    fig.savefig(paths["plot_pdf"])
    plt.close(fig)


def write_reports(
    paths: dict[str, Path],
    input_path: Path,
    reference_path: Path,
    columns: list[str],
    total: int,
    missing_counts: collections.Counter[str],
    taxon_counts: dict[str, collections.Counter[str]],
    reference_taxa: dict[str, set[str]],
) -> None:
    taxonomy = {}
    for rank in TAXONOMY_COLUMNS:
        taxonomy[rank] = {
            **distribution_summary(taxon_counts[rank]),
            "missing_count": missing_counts[rank],
            "missing_percent": 100 * missing_counts[rank] / total,
            "bioscan5m_overlap": overlap_summary(
                taxon_counts[rank], reference_taxa[rank]
            ),
        }
    missingness = {
        column: {
            "missing_count": missing_counts[column],
            "missing_percent": 100 * missing_counts[column] / total,
        }
        for column in columns
    }
    payload = {
        "input_csv": str(input_path.resolve()),
        "bioscan5m_reference_csv": str(reference_path.resolve()),
        "marker_filter": MARKER,
        "coi5p_rows": total,
        "missing_value_definition": "empty string after stripping whitespace",
        "overlap_definition": (
            "exact case-sensitive taxonomy label match after stripping whitespace; "
            "blank labels excluded; overlapping_samples counts new COI-5P samples "
            "whose label occurs in BIOSCAN-5M"
        ),
        "taxonomy": taxonomy,
        "column_missingness": missingness,
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# COI-5P taxonomy statistics",
        "",
        f"- Input: `{input_path.resolve()}`",
        f"- BIOSCAN-5M reference: `{reference_path.resolve()}`",
        f"- Filter: `marker_code == {MARKER}`",
        f"- COI-5P records: {total:,}",
        "- Missing-value definition: empty string after stripping whitespace",
        "",
        "## Taxonomy summary",
        "",
        "| Rank | Unique taxa | Missing | Missing % | Median records/taxon | Max records/taxon | Singletons |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank in TAXONOMY_COLUMNS:
        item = taxonomy[rank]
        lines.append(
            f"| {rank} | {item['unique_taxa']:,} | {item['missing_count']:,} | "
            f"{item['missing_percent']:.4f}% | {item['median_records_per_taxon']:,.1f} | "
            f"{item['maximum_records_per_taxon']:,} | {item['singleton_taxa']:,} |"
        )
    lines.extend(
        [
            "",
            "## Exact taxonomy overlap with BIOSCAN-5M",
            "",
            "Class overlap uses exact, case-sensitive taxonomy labels after stripping whitespace; blank labels are excluded. Class overlap ratio = overlapping classes / nonmissing classes in the new data. Sample overlap ratio = new COI-5P samples whose label occurs in BIOSCAN-5M / samples with a nonmissing label at that rank in the new data.",
            "",
            "| Rank | New classes | BIOSCAN-5M classes | Overlapping classes | Class overlap (overlap / new) | New nonmissing samples | Overlapping samples | Sample overlap (overlap / new) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank in TAXONOMY_COLUMNS:
        overlap = taxonomy[rank]["bioscan5m_overlap"]
        lines.append(
            f"| {rank} | {taxonomy[rank]['unique_taxa']:,} | "
            f"{overlap['bioscan5m_unique_taxa']:,} | "
            f"{overlap['overlapping_taxa']:,} | "
            f"{overlap['overlapping_taxa']:,} / {taxonomy[rank]['unique_taxa']:,} "
            f"({overlap['new_taxa_overlap_percent']:.4f}%) | "
            f"{overlap['nonmissing_samples']:,} | "
            f"{overlap['overlapping_samples']:,} | "
            f"{overlap['overlapping_samples']:,} / {overlap['nonmissing_samples']:,} "
            f"({overlap['nonmissing_sample_overlap_percent']:.4f}%) |"
        )
    lines.extend(
        [
            "",
            "## Column missingness",
            "",
            "| Column | Missing | Missing % |",
            "|---|---:|---:|",
        ]
    )
    for column in columns:
        item = missingness[column]
        lines.append(
            f"| {column} | {item['missing_count']:,} | {item['missing_percent']:.4f}% |"
        )
    lines.append("")
    paths["report"].write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve(strict=True)
    reference_path = args.reference_csv.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    paths = output_paths(output_dir)
    validate_outputs(paths, args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    columns, total, missing_counts, taxon_counts = scan_csv(input_path)
    reference_taxa = scan_reference_taxa(reference_path)
    write_csv_outputs(
        paths, columns, total, missing_counts, taxon_counts, reference_taxa
    )
    write_plot(paths, taxon_counts)
    write_reports(
        paths,
        input_path,
        reference_path,
        columns,
        total,
        missing_counts,
        taxon_counts,
        reference_taxa,
    )

    print(f"COI-5P rows: {total:,}")
    for rank in TAXONOMY_COLUMNS:
        print(
            f"{rank}: unique={len(taxon_counts[rank]):,}, "
            f"missing={missing_counts[rank]:,}, "
            f"overlap={overlap_summary(taxon_counts[rank], reference_taxa[rank])['overlapping_taxa']:,}"
        )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
