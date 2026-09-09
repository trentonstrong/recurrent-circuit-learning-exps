#!/usr/bin/env python3
"""Command-line entry point for R000 acceptance operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recurrent_circuit_learning.r000 import (
    checkpoint_resume_check,
    export_fixture,
    profile_update,
    verify_fixture,
    verify_source,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_parser = subparsers.add_parser("verify-source")
    source_parser.add_argument("path", type=Path)

    export_parser = subparsers.add_parser("export-fixture")
    export_parser.add_argument("path", type=Path)

    verify_parser = subparsers.add_parser("verify-fixture")
    verify_parser.add_argument("path", type=Path)
    verify_parser.add_argument("--report", type=Path)

    checkpoint_parser = subparsers.add_parser("checkpoint-check")
    checkpoint_parser.add_argument("path", type=Path)
    checkpoint_parser.add_argument("--report", type=Path)

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--repetitions", type=int, default=3)
    profile_parser.add_argument("--report", type=Path)

    args = parser.parse_args()
    if args.command == "verify-source":
        result = verify_source(args.path)
    elif args.command == "export-fixture":
        result = export_fixture(args.path)
    elif args.command == "verify-fixture":
        result = verify_fixture(args.path, args.report)
    elif args.command == "checkpoint-check":
        result = checkpoint_resume_check(args.path)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        result = profile_update(args.repetitions)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
