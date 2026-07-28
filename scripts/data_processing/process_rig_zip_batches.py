#!/usr/bin/env python3
"""Resumable, safety-first ZIP copy, batch extraction, and transfer workflow.

Run ``inspect_rig_zip_and_build_batches.py`` first with the same local
``state`` directory. This program intentionally requires all machine-specific
paths on the command line and never deletes the source ZIP or final output.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

# These values are populated by configure() immediately after argument parsing.
SOURCE_ZIP: Path
LOCAL_ROOT: Path
LOCAL_ZIP: Path
LOCAL_BATCH_ROOT: Path
LOG_DIR: Path
STATE_DIR: Path
MANIFEST: Path
SHARED_OUTPUT_ROOT: Path
COMMON_ROOT: str
MARGIN_BYTES: int
LOCK_PATH: Path
DEST_MARKER: Path
VERIFY_MARKER: Path
VALID_STATUSES = {"pending", "extracting", "extracted", "transferring", "verified", "cleaned", "failed"}


class WorkflowError(RuntimeError):
    pass


def configure(args: argparse.Namespace) -> None:
    """Resolve CLI paths and derive every workflow-owned file location."""
    global SOURCE_ZIP, LOCAL_ROOT, LOCAL_ZIP, LOCAL_BATCH_ROOT
    global LOG_DIR, STATE_DIR, MANIFEST, SHARED_OUTPUT_ROOT, COMMON_ROOT
    global MARGIN_BYTES, LOCK_PATH, DEST_MARKER, VERIFY_MARKER

    SOURCE_ZIP = args.source_zip.expanduser().resolve()
    LOCAL_ROOT = args.local_root.expanduser().resolve()
    LOCAL_ZIP = (
        args.local_zip.expanduser().resolve()
        if args.local_zip is not None
        else LOCAL_ROOT / SOURCE_ZIP.name
    )
    LOCAL_BATCH_ROOT = LOCAL_ROOT / "extract_batches"
    LOG_DIR = LOCAL_ROOT / "logs"
    STATE_DIR = LOCAL_ROOT / "state"
    MANIFEST = STATE_DIR / "batches.tsv"
    SHARED_OUTPUT_ROOT = args.shared_output_root.expanduser().resolve()
    COMMON_ROOT = args.common_root
    MARGIN_BYTES = int(args.margin_gib * 1024**3)
    LOCK_PATH = STATE_DIR / "process_batches.lock"
    DEST_MARKER = SHARED_OUTPUT_ROOT / ".bioscan_extract_workflow.json"
    VERIFY_MARKER = STATE_DIR / "local_zip.verified.json"

    if not COMMON_ROOT or "/" in COMMON_ROOT or "\\" in COMMON_ROOT:
        raise WorkflowError("--common-root must be one top-level directory name")
    if MARGIN_BYTES < 0:
        raise WorkflowError("--margin-gib must be non-negative")
    if LOCAL_ROOT == SHARED_OUTPUT_ROOT:
        raise WorkflowError("--local-root and --shared-output-root must differ")


def setup() -> None:
    for path in (LOCAL_ROOT, LOCAL_BATCH_ROOT, LOG_DIR, STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def resolved(path: Path, strict: bool = False) -> Path:
    return path.expanduser().resolve(strict=strict)


def print_config() -> None:
    values = {
        "SOURCE_ZIP": resolved(SOURCE_ZIP, SOURCE_ZIP.exists()),
        "LOCAL_ROOT": resolved(LOCAL_ROOT),
        "LOCAL_ZIP": resolved(LOCAL_ZIP),
        "LOCAL_BATCH_ROOT": resolved(LOCAL_BATCH_ROOT),
        "LOG_DIR": resolved(LOG_DIR),
        "STATE_DIR": resolved(STATE_DIR),
        "SHARED_OUTPUT_ROOT": resolved(SHARED_OUTPUT_ROOT),
        "SAFETY_MARGIN_BYTES": MARGIN_BYTES,
    }
    for key, value in values.items():
        print(f"{key}={value}")


class Lock:
    def __enter__(self):
        self.handle = LOCK_PATH.open("a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkflowError(f"Another process holds {LOCK_PATH}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} started={time.time()}\n")
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


def read_manifest() -> tuple[list[dict[str, str]], list[str]]:
    if not MANIFEST.is_file():
        raise WorkflowError(f"Manifest is missing: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fields = reader.fieldnames or []
    expected = ["batch_id", "selection_type", "pattern_or_filelist", "estimated_uncompressed_bytes", "status"]
    if fields != expected:
        raise WorkflowError(f"Unexpected manifest columns: {fields}")
    for row in rows:
        if row["status"] not in VALID_STATUSES:
            raise WorkflowError(f"Invalid status for {row['batch_id']}: {row['status']}")
    return rows, fields


def write_manifest(rows: list[dict[str, str]], fields: list[str]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix="batches.", suffix=".tmp", dir=STATE_DIR)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, MANIFEST)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def set_status(batch_id: str, status: str) -> None:
    if status not in VALID_STATUSES:
        raise WorkflowError(f"Invalid status: {status}")
    rows, fields = read_manifest()
    for row in rows:
        if row["batch_id"] == batch_id:
            old = row["status"]
            row["status"] = status
            write_manifest(rows, fields)
            logging.info("Status %s: %s -> %s", batch_id, old, status)
            return
    raise WorkflowError(f"Unknown batch: {batch_id}")


def run_logged(command: list[str], log_path: Path, *, require_no_output: bool = False) -> None:
    logging.info("Command: %s", " ".join(repr(item) for item in command))
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] command={command!r}\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        output_seen = False
        assert process.stdout is not None
        for line in process.stdout:
            output_seen = output_seen or bool(line.strip())
            sys.stdout.write(line)
            log.write(line)
        result = process.wait()
    if result != 0:
        raise WorkflowError(f"Command exited {result}; see {log_path}")
    if require_no_output and output_seen:
        raise WorkflowError(f"Verification reported differences; see {log_path}")


def atomic_write_text(path: Path, content: str) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def free_bytes() -> int:
    return shutil.disk_usage(LOCAL_ROOT).free


def require_free(required: int, context: str) -> None:
    available = free_bytes()
    logging.info("Local free bytes before %s: %d (required: %d)", context, available, required)
    if available < required:
        raise WorkflowError(f"Insufficient local space for {context}: free={available}, required={required}")


def safe_member(name: str) -> PurePosixPath:
    if not name or "\x00" in name:
        raise WorkflowError(f"Unsafe empty/NUL member name: {name!r}")
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized.startswith("/") or (len(normalized) >= 2 and normalized[1] == ":"):
        raise WorkflowError(f"Unsafe absolute member: {name!r}")
    if ".." in path.parts:
        raise WorkflowError(f"Unsafe parent traversal member: {name!r}")
    if not path.parts or path.parts[0] != COMMON_ROOT:
        raise WorkflowError(f"Member is outside expected common root: {name!r}")
    return path


def load_members(row: dict[str, str]) -> list[dict]:
    path = Path(row["pattern_or_filelist"])
    if row["selection_type"] != "member_index_jsonl" or not path.is_file():
        raise WorkflowError(f"Invalid member list for {row['batch_id']}: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def copy_zip() -> None:
    if not SOURCE_ZIP.is_file() or not os.access(SOURCE_ZIP, os.R_OK):
        raise WorkflowError(f"Source is not a readable file: {SOURCE_ZIP}")
    source_size = SOURCE_ZIP.stat().st_size
    if LOCAL_ZIP.exists() and not LOCAL_ZIP.is_file():
        raise WorkflowError(f"Local ZIP path exists but is not a file: {LOCAL_ZIP}")
    existing = LOCAL_ZIP.stat().st_size if LOCAL_ZIP.exists() else 0
    if existing > source_size:
        raise WorkflowError(f"Local ZIP is larger than source; refusing overwrite: {LOCAL_ZIP}")
    require_free((source_size - existing) + MARGIN_BYTES, "ZIP copy")
    run_logged(
        ["rsync", "-ah", "--dry-run", "--itemize-changes", str(SOURCE_ZIP), str(LOCAL_ZIP)],
        LOG_DIR / "copy_zip_dry_run.log",
    )
    run_logged(
        ["rsync", "-ah", "--partial", "--append-verify", "--info=progress2", str(SOURCE_ZIP), str(LOCAL_ZIP)],
        LOG_DIR / "copy_zip.log",
    )
    if LOCAL_ZIP.stat().st_size != source_size:
        raise WorkflowError("Source/local ZIP byte sizes differ after rsync")
    logging.info("ZIP copy byte-size check passed: %d", source_size)


def sha256(path: Path, output_path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024**2):
            digest.update(chunk)
    value = digest.hexdigest()
    output_path.write_text(f"{value}  {path}\n", encoding="utf-8")
    return value


def verify_zip(method: str) -> None:
    if not SOURCE_ZIP.is_file() or not LOCAL_ZIP.is_file():
        raise WorkflowError("Both source and local ZIP must exist")
    if SOURCE_ZIP.stat().st_size != LOCAL_ZIP.stat().st_size:
        raise WorkflowError("Source/local ZIP byte sizes differ")
    details = {"method": method, "bytes": LOCAL_ZIP.stat().st_size, "verified_at": time.time()}
    if method == "sha256":
        source_hash = sha256(SOURCE_ZIP, STATE_DIR / "source_zip.sha256")
        local_hash = sha256(LOCAL_ZIP, STATE_DIR / "local_zip.sha256")
        if source_hash != local_hash:
            raise WorkflowError("Source/local SHA-256 values differ")
        details["sha256"] = local_hash
    else:
        run_logged(["unzip", "-t", str(LOCAL_ZIP)], LOG_DIR / "test_local_zip.log")
    temp = VERIFY_MARKER.with_suffix(".tmp")
    temp.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, VERIFY_MARKER)
    logging.info("Local ZIP verification passed using %s", method)


def validate_local_zip() -> None:
    if not LOCAL_ZIP.is_file() or not VERIFY_MARKER.is_file():
        raise WorkflowError(f"Copy and verify local ZIP first; missing {LOCAL_ZIP} or {VERIFY_MARKER}")
    details = json.loads(VERIFY_MARKER.read_text(encoding="utf-8"))
    if LOCAL_ZIP.stat().st_size != details.get("bytes") or SOURCE_ZIP.stat().st_size != details.get("bytes"):
        raise WorkflowError("ZIP size changed since verification")


def extract_batch(row: dict[str, str], archive: zipfile.ZipFile) -> dict:
    batch_id = row["batch_id"]
    members = load_members(row)
    expected = int(row["estimated_uncompressed_bytes"])
    require_free(expected + MARGIN_BYTES, f"extracting {batch_id}")
    batch_dir = resolved(LOCAL_BATCH_ROOT / batch_id)
    root = resolved(LOCAL_BATCH_ROOT)
    if batch_dir.parent != root:
        raise WorkflowError(f"Batch path escaped root: {batch_dir}")
    batch_dir.mkdir(parents=True, exist_ok=True)
    infos = archive.infolist()
    set_status(batch_id, "extracting")
    written = skipped = files = 0
    last_space_check = 0
    log_path = LOG_DIR / f"extract_{batch_id}.log"
    with log_path.open("a", encoding="utf-8") as log:
        for position, member in enumerate(members, 1):
            index = int(member["index"])
            if index < 0 or index >= len(infos):
                raise WorkflowError(f"Member index out of range: {index}")
            info = infos[index]
            if (info.filename, info.file_size, info.CRC, info.is_dir()) != (
                member["name"], int(member["size"]), int(member["crc"]), bool(member["is_dir"])
            ):
                raise WorkflowError(f"ZIP member metadata changed at index {index}")
            relative = safe_member(info.filename)
            target = batch_dir.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            files += 1
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.is_file() and target.stat().st_size == info.file_size:
                    skipped += info.file_size
                    continue
                raise WorkflowError(f"Existing extracted path has wrong type/size: {target}")
            part = target.with_name(target.name + ".part")
            if part.exists():
                if not part.is_file():
                    raise WorkflowError(f"Partial path is not a file: {part}")
                part.unlink()
            with archive.open(info, "r") as source, part.open("xb") as destination:
                while chunk := source.read(8 * 1024**2):
                    destination.write(chunk)
                    written += len(chunk)
                    if written - last_space_check >= 1024**3:
                        require_free(MARGIN_BYTES, f"streaming {batch_id}")
                        last_space_check = written
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(part, target)
            timestamp = time.mktime(info.date_time + (0, 0, -1))
            os.utime(target, (timestamp, timestamp))
            if position % 1000 == 0:
                message = f"progress members={position}/{len(members)} written={written} skipped={skipped}"
                logging.info("%s %s", batch_id, message)
                log.write(message + "\n")
                log.flush()
    actual_files = sum(1 for path in batch_dir.rglob("*") if path.is_file() and not path.name.endswith(".part"))
    actual_bytes = sum(path.stat().st_size for path in batch_dir.rglob("*") if path.is_file() and not path.name.endswith(".part"))
    if actual_files != files or actual_bytes != expected:
        raise WorkflowError(
            f"Extraction totals mismatch for {batch_id}: files={actual_files}/{files}, bytes={actual_bytes}/{expected}"
        )
    stats = {"batch_id": batch_id, "file_count": actual_files, "bytes": actual_bytes, "written": written, "skipped": skipped}
    (STATE_DIR / f"{batch_id}.extracted.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    set_status(batch_id, "extracted")
    return stats


def destination_is_owned() -> bool:
    if not DEST_MARKER.is_file():
        return False
    try:
        marker = json.loads(DEST_MARKER.read_text(encoding="utf-8"))
        return marker.get("source_zip") == str(resolved(SOURCE_ZIP, True)) and marker.get("common_root") == COMMON_ROOT
    except (OSError, json.JSONDecodeError):
        return False


def prepare_destination() -> None:
    if SHARED_OUTPUT_ROOT.exists():
        if not SHARED_OUTPUT_ROOT.is_dir() or not destination_is_owned():
            raise WorkflowError(f"Shared destination exists without the expected workflow marker: {SHARED_OUTPUT_ROOT}")
        return
    parent = SHARED_OUTPUT_ROOT.parent
    if not parent.is_dir():
        raise WorkflowError(f"Shared destination parent is missing: {parent}")
    SHARED_OUTPUT_ROOT.mkdir()
    marker = {"source_zip": str(resolved(SOURCE_ZIP, True)), "common_root": COMMON_ROOT, "created_at": time.time()}
    DEST_MARKER.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def verify_batch_destination(row: dict[str, str], source_dir: Path) -> dict:
    members = load_members(row)
    expected_files = 0
    expected_bytes = 0
    for member in members:
        relative = safe_member(member["name"])
        if member["is_dir"]:
            continue
        expected_files += 1
        expected_bytes += int(member["size"])
        target = SHARED_OUTPUT_ROOT.joinpath(*relative.parts[1:])
        if not target.is_file() or target.stat().st_size != int(member["size"]):
            raise WorkflowError(f"Missing or wrong-size shared file: {target}")
    log_path = LOG_DIR / f"verify_{row['batch_id']}.log"
    run_logged(
        ["rsync", "-a", "--omit-dir-times", "--dry-run", "--checksum", "--itemize-changes", str(source_dir) + "/", str(SHARED_OUTPUT_ROOT) + "/"],
        log_path,
        require_no_output=True,
    )
    stats = {"batch_id": row["batch_id"], "file_count": expected_files, "bytes": expected_bytes, "verified_at": time.time()}
    (STATE_DIR / f"{row['batch_id']}.verified.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def transfer_batch(row: dict[str, str]) -> None:
    batch_id = row["batch_id"]
    batch_dir = resolved(LOCAL_BATCH_ROOT / batch_id)
    source_dir = batch_dir / COMMON_ROOT
    if not source_dir.is_dir():
        raise WorkflowError(f"Extracted common-root directory is missing: {source_dir}")
    dry_log = LOG_DIR / f"rsync_{batch_id}_dry_run.log"
    run_logged(
        ["rsync", "-a", "--omit-dir-times", "--dry-run", "--ignore-existing", "--itemize-changes", str(source_dir) + "/", str(SHARED_OUTPUT_ROOT) + "/"],
        dry_log,
    )
    prepare_destination()
    set_status(batch_id, "transferring")
    run_logged(
        ["rsync", "-a", "--omit-dir-times", "--partial-dir=.rsync-partial-" + batch_id, "--ignore-existing", "--info=progress2", str(source_dir) + "/", str(SHARED_OUTPUT_ROOT) + "/"],
        LOG_DIR / f"rsync_{batch_id}.log",
    )
    verify_batch_destination(row, source_dir)
    set_status(batch_id, "verified")


def cleanup_batch(row: dict[str, str]) -> None:
    batch_id = row["batch_id"]
    rows, _ = read_manifest()
    current = next(item for item in rows if item["batch_id"] == batch_id)
    if current["status"] != "verified":
        raise WorkflowError(f"Refusing cleanup because {batch_id} is not verified")
    root = resolved(LOCAL_BATCH_ROOT)
    target = resolved(LOCAL_BATCH_ROOT / batch_id)
    forbidden = {Path("/"), resolved(Path.home()), resolved(LOCAL_ROOT), root}
    print(f"Deleting verified local batch: {target!s}")
    print(f"Resolved deletion path: {target!s}")
    print(f"Deletion parent root: {root!s}")
    if not str(target) or target in forbidden or target.parent != root or not target.name.startswith("batch_"):
        raise WorkflowError(f"Unsafe cleanup target: {target}")
    if target.exists():
        size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
        print(f"Deletion target bytes: {size}")
        shutil.rmtree(target)
    set_status(batch_id, "cleaned")


def process(retry_failed: bool) -> None:
    validate_local_zip()
    rows, _ = read_manifest()
    with zipfile.ZipFile(LOCAL_ZIP) as archive:
        for original in rows:
            batch_id = original["batch_id"]
            row = next(item for item in read_manifest()[0] if item["batch_id"] == batch_id)
            status = row["status"]
            if status in {"cleaned", "verified"}:
                logging.info("Skipping %s with status %s", batch_id, status)
                continue
            if status == "failed" and not retry_failed:
                raise WorkflowError(f"{batch_id} is failed; inspect logs, then use run --retry-failed")
            try:
                if status in {"pending", "extracting", "failed"}:
                    extract_batch(row, archive)
                row = next(item for item in read_manifest()[0] if item["batch_id"] == batch_id)
                if row["status"] in {"extracted", "transferring"}:
                    transfer_batch(row)
                row = next(item for item in read_manifest()[0] if item["batch_id"] == batch_id)
                if row["status"] == "verified":
                    cleanup_batch(row)
            except Exception:
                set_status(batch_id, "failed")
                logging.exception("Batch failed and local files were preserved: %s", batch_id)
                raise


def status() -> None:
    rows, _ = read_manifest()
    counts = {name: 0 for name in sorted(VALID_STATUSES)}
    for row in rows:
        counts[row["status"]] += 1
    extracted = transferred = 0
    current = []
    for row in rows:
        extract_stats = STATE_DIR / f"{row['batch_id']}.extracted.json"
        verify_stats = STATE_DIR / f"{row['batch_id']}.verified.json"
        if extract_stats.is_file():
            extracted += json.loads(extract_stats.read_text())["bytes"]
        if verify_stats.is_file():
            transferred += json.loads(verify_stats.read_text())["bytes"]
        if row["status"] not in {"pending", "cleaned"}:
            current.append(f"{row['batch_id']}:{row['status']}")
    print(f"total_batches={len(rows)}")
    for name, count in counts.items():
        print(f"{name}={count}")
    print(f"current={','.join(current) or 'none'}")
    print(f"bytes_extracted={extracted}")
    print(f"bytes_transferred_verified={transferred}")
    print(f"local_free_bytes={free_bytes()}")
    print(f"log_dir={resolved(LOG_DIR)}")
    print(f"manifest={resolved(MANIFEST, True)}")


def final_verify() -> None:
    rows, _ = read_manifest()
    incomplete = [f"{row['batch_id']}:{row['status']}" for row in rows if row["status"] not in {"verified", "cleaned"}]
    if incomplete:
        raise WorkflowError(f"Incomplete batches: {', '.join(incomplete)}")
    if not destination_is_owned():
        raise WorkflowError("Shared destination workflow marker is missing or invalid")
    expected_count = expected_bytes = 0
    expected_names = set()
    for row in rows:
        for member in load_members(row):
            relative = safe_member(member["name"])
            if member["is_dir"]:
                continue
            rel = Path(*relative.parts[1:])
            expected_names.add(rel.as_posix())
            expected_count += 1
            expected_bytes += int(member["size"])
    actual_count = actual_bytes = 0
    unexpected = []
    for path in SHARED_OUTPUT_ROOT.rglob("*"):
        if not path.is_file() or path == DEST_MARKER or ".rsync-partial-" in path.as_posix():
            continue
        rel = path.relative_to(SHARED_OUTPUT_ROOT).as_posix()
        actual_count += 1
        actual_bytes += path.stat().st_size
        if rel not in expected_names and len(unexpected) < 100:
            unexpected.append(rel)
    missing = []
    for name in expected_names:
        if not (SHARED_OUTPUT_ROOT / name).is_file() and len(missing) < 100:
            missing.append(name)
    report = {
        "expected_files": expected_count,
        "actual_files": actual_count,
        "expected_bytes": expected_bytes,
        "actual_bytes": actual_bytes,
        "missing_sample": missing,
        "unexpected_sample": unexpected,
        "passed": expected_count == actual_count and expected_bytes == actual_bytes and not missing and not unexpected,
        "verified_at": time.time(),
        "local_zip_retained": str(resolved(LOCAL_ZIP, True)),
    }
    path = STATE_DIR / "final_report.txt"
    atomic_write_text(path, json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise WorkflowError(f"Final verification failed; see {path}")
    print(f"Local ZIP was NOT deleted. Optional manual command after review: rm -- {str(LOCAL_ZIP)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a ZIP to local scratch, extract manifest-defined batches, "
            "rsync them to shared storage, and verify before local cleanup."
        )
    )
    parser.add_argument(
        "--source-zip",
        type=Path,
        required=True,
        help="read-only source ZIP",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        required=True,
        help="scratch directory containing state, logs, batches, and local ZIP",
    )
    parser.add_argument(
        "--local-zip",
        type=Path,
        help="local ZIP path (default: <local-root>/<source ZIP filename>)",
    )
    parser.add_argument(
        "--shared-output-root",
        type=Path,
        required=True,
        help="final extracted directory",
    )
    parser.add_argument(
        "--common-root",
        required=True,
        help="single top-level directory reported by archive inspection",
    )
    parser.add_argument(
        "--margin-gib",
        type=float,
        default=100.0,
        help="free-space reserve required during extraction (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="print the resolved workflow configuration")
    sub.add_parser("status", help="show manifest and byte progress")
    sub.add_parser("copy-zip", help="copy the source ZIP to local scratch")
    verify = sub.add_parser("verify-zip", help="verify the local ZIP copy")
    verify.add_argument("--method", choices=("sha256", "unzip-test"), default="sha256")
    run = sub.add_parser("run", help="extract, transfer, verify, and clean batches")
    run.add_argument("--retry-failed", action="store_true")
    sub.add_parser("final-verify", help="verify all files in the final directory")
    args = parser.parse_args()
    configure(args)
    setup()
    if args.command == "config":
        print_config()
    elif args.command == "status":
        status()
    else:
        print_config()
        with Lock():
            if args.command == "copy-zip":
                copy_zip()
            elif args.command == "verify-zip":
                verify_zip(args.method)
            elif args.command == "run":
                process(args.retry_failed)
            elif args.command == "final-verify":
                final_verify()


if __name__ == "__main__":
    try:
        main()
    except (WorkflowError, OSError, zipfile.BadZipFile, subprocess.SubprocessError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1)
