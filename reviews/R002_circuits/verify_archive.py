#!/usr/bin/env python3
"""Verify the six-run R002 upload against committed manifests, then extract it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = [
    "R002_formal_attempt01_seed04_categorical_reference_decay",
    "R002_formal_attempt01_seed13_categorical_reference_decay",
    "R002_formal_attempt01_seed04_categorical_no_decay",
    "R002_formal_attempt01_seed15_categorical_no_decay",
    "R002_formal_attempt01_seed08_truth_reference_decay",
    "R002_formal_attempt01_seed14_truth_reference_decay",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    expected = {}
    for run in RUNS:
        manifest = json.loads((ROOT / "results" / run / "manifest.json").read_text())
        entries = list(manifest["artifacts"])
        for checkpoint in manifest["checkpoints"]:
            entries.extend((checkpoint["checkpoint"], checkpoint["probe_trajectory"]))
        for entry in entries:
            path = str(PurePosixPath("artifacts") / run / PurePosixPath(entry["path"]).name)
            if path in expected:
                raise ValueError(f"Duplicate manifest entry: {path}")
            expected[path] = entry
    if len(expected) != 144:
        raise ValueError(f"Expected 144 files, found {len(expected)} manifest entries")

    payloads = {}
    verified = []
    allowed_directories = {f"artifacts/{run}" for run in RUNS}
    with tarfile.open(args.archive, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isdir() and member.name.rstrip("/") in allowed_directories:
                continue
            if not member.isfile() or member.name not in expected or member.name in payloads:
                raise ValueError(f"Unexpected archive member: {member.name}")
            entry = expected[member.name]
            if member.size != entry["bytes"]:
                raise ValueError(f"Length mismatch: {member.name}")
            with archive.extractfile(member) as source:
                data = source.read()
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry["sha256"] or len(data) != entry["bytes"]:
                raise ValueError(f"Payload mismatch: {member.name}")
            payloads[member.name] = data
            verified.append({"path": member.name, "bytes": len(data), "sha256": digest})
    if set(payloads) != set(expected):
        raise ValueError(f"Missing payloads: {sorted(set(expected) - set(payloads))}")

    # Validate every destination before writing; never overwrite a different file.
    destinations = {}
    for name, data in payloads.items():
        destination = (ROOT / name).resolve()
        if not destination.is_relative_to(ROOT.resolve()):
            raise ValueError(f"Destination escapes repository: {name}")
        if destination.exists() and destination.read_bytes() != data:
            raise ValueError(f"Existing destination differs: {name}")
        destinations[name] = destination
    for name, destination in destinations.items():
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payloads[name])

    report = {
        "archive": {
            "name": args.archive.name,
            "bytes": args.archive.stat().st_size,
            "sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        },
        "verified_files": verified,
    }
    (OUT / "artifact_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(verified)} files; {sum(len(x) for x in payloads.values()):,} payload bytes")


if __name__ == "__main__":
    main()
