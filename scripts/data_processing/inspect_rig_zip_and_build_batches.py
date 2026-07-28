#!/usr/bin/env python3
"""Inspect a ZIP safely and create byte-bounded extraction batch manifests.

The archive is read but never extracted or modified. Unsafe paths, symlinks,
and archive structure are checked before resumable batch state is written.
"""

import argparse
import collections
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

GIB = 1024 ** 3


def unsafe_reason(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not name or "\x00" in name:
        return "empty_or_nul"
    if path.is_absolute() or normalized.startswith("/"):
        return "absolute"
    if len(normalized) >= 2 and normalized[1] == ":":
        return "drive_path"
    if ".." in path.parts:
        return "parent_component"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a ZIP without extracting it and write resumable batch "
            "manifests bounded by estimated uncompressed size."
        )
    )
    parser.add_argument("--zip", required=True, type=Path, dest="zip_path")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument(
        "--target-gib",
        type=float,
        default=150.0,
        help="maximum estimated uncompressed bytes per batch in GiB",
    )
    args = parser.parse_args()

    if args.target_gib <= 0:
        parser.error("--target-gib must be positive")

    state = args.state_dir.expanduser().resolve()
    state.mkdir(parents=True, exist_ok=True)
    target = int(args.target_gib * GIB)

    zip_path = args.zip_path.expanduser().resolve(strict=True)
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()

    names = collections.Counter(info.filename for info in infos)
    unsafe = []
    symlinks = []
    top_counts = collections.Counter()
    top_sizes = collections.Counter()
    second_counts = collections.Counter()
    second_sizes = collections.Counter()
    extensions = collections.Counter()
    extension_sizes = collections.Counter()
    depths = collections.Counter()
    total_bytes = total_compressed = file_count = directory_count = 0

    for index, info in enumerate(infos):
        reason = unsafe_reason(info.filename)
        if reason:
            unsafe.append({"index": index, "name": info.filename, "reason": reason})
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            symlinks.append({"index": index, "name": info.filename})
        parts = PurePosixPath(info.filename.replace("\\", "/")).parts
        if parts:
            top_counts[parts[0]] += 1
            top_sizes[parts[0]] += info.file_size
        if len(parts) >= 2:
            key = "/".join(parts[:2])
            second_counts[key] += 1
            second_sizes[key] += info.file_size
        depths[len(parts)] += 1
        total_bytes += info.file_size
        total_compressed += info.compress_size
        if info.is_dir():
            directory_count += 1
        else:
            file_count += 1
            suffix = PurePosixPath(info.filename).suffix.lower() or "<none>"
            extensions[suffix] += 1
            extension_sizes[suffix] += info.file_size

    common_roots = list(top_counts)
    duplicate_name_count = sum(1 for count in names.values() if count > 1)
    if unsafe or symlinks or duplicate_name_count:
        rejection_path = state / "inspection_rejection.json"
        rejection_path.write_text(
            json.dumps(
                {
                    "archive": str(zip_path),
                    "unsafe_count": len(unsafe),
                    "unsafe_entries": unsafe[:100],
                    "symlink_count": len(symlinks),
                    "symlinks": symlinks[:100],
                    "duplicate_name_count": duplicate_name_count,
                    "duplicate_names_sample": [
                        name for name, count in names.items() if count > 1
                    ][:100],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise SystemExit(
            "Refusing to create batches: "
            f"unsafe={len(unsafe)} symlinks={len(symlinks)} "
            f"duplicate_names={duplicate_name_count}; inspect {rejection_path}"
        )

    # Archive-order batches preserve exact ZipInfo indexes and stay under a byte target.
    batches: list[list[tuple[int, zipfile.ZipInfo]]] = []
    current: list[tuple[int, zipfile.ZipInfo]] = []
    current_bytes = 0
    for index, info in enumerate(infos):
        if current and current_bytes + info.file_size > target:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append((index, info))
        current_bytes += info.file_size
    if current:
        batches.append(current)

    manifest_path = state / "batches.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest:
        manifest.write("batch_id\tselection_type\tpattern_or_filelist\testimated_uncompressed_bytes\tstatus\n")
        for number, batch in enumerate(batches, 1):
            batch_id = f"batch_{number:03d}"
            members_path = state / f"{batch_id}.members.jsonl"
            estimated = 0
            with members_path.open("w", encoding="utf-8") as members:
                for index, info in batch:
                    estimated += info.file_size
                    members.write(json.dumps({
                        "index": index,
                        "name": info.filename,
                        "size": info.file_size,
                        "crc": info.CRC,
                        "is_dir": info.is_dir(),
                    }, ensure_ascii=True) + "\n")
            manifest.write(
                f"{batch_id}\tmember_index_jsonl\t{members_path}\t{estimated}\tpending\n"
            )

    report = {
        "archive": str(zip_path),
        "entry_count": len(infos),
        "file_count": file_count,
        "directory_count": directory_count,
        "uncompressed_bytes": total_bytes,
        "compressed_member_bytes": total_compressed,
        "zip_file_bytes": os.path.getsize(zip_path),
        "common_root": common_roots[0] if len(common_roots) == 1 else None,
        "top_level": [
            {"name": key, "entries": top_counts[key], "bytes": top_sizes[key]}
            for key in sorted(top_counts)
        ],
        "second_level_largest": [
            {"name": key, "entries": second_counts[key], "bytes": second_sizes[key]}
            for key in sorted(second_counts, key=second_sizes.get, reverse=True)[:100]
        ],
        "path_depth_counts": dict(sorted(depths.items())),
        "extensions": [
            {"extension": key, "files": extensions[key], "bytes": extension_sizes[key]}
            for key in sorted(extensions, key=extension_sizes.get, reverse=True)
        ],
        "duplicate_name_count": duplicate_name_count,
        "duplicate_entry_count": sum(count - 1 for count in names.values() if count > 1),
        "duplicate_names_sample": [name for name, count in names.items() if count > 1][:100],
        "unsafe_count": len(unsafe),
        "unsafe_entries": unsafe[:100],
        "symlink_count": len(symlinks),
        "symlinks": symlinks[:100],
        "batch_target_bytes": target,
        "batch_count": len(batches),
    }
    (state / "inspection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (state / "top_level_entry_counts.txt").open("w", encoding="utf-8") as handle:
        for key in sorted(top_counts, key=top_counts.get, reverse=True):
            handle.write(f"{top_counts[key]:10d} {top_sizes[key]:15d} {key}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
