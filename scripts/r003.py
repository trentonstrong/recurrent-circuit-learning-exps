#!/usr/bin/env python3
"""R003 preflight and fixed-cohort execution entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Must be set before importing JAX.
os.environ.setdefault("XLA_FLAGS", " --xla_gpu_deterministic_ops=true")

from recurrent_circuit_learning.r003 import run_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "run"):
        current = subparsers.add_parser(command)
        current.add_argument("--config", type=Path, required=True)
        current.add_argument("--run-id", required=True)
        current.add_argument("--repository-root", type=Path, default=Path.cwd())
        current.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
        current.add_argument("--results-root", type=Path, default=Path("results"))
        current.add_argument("--gpu-lock", type=Path, default=Path("envs/jax-gpu/uv.lock"))
    args = parser.parse_args()
    report = run_diagnostic(
        args.config,
        args.run_id,
        args.repository_root.resolve(),
        args.artifact_root.resolve(),
        args.results_root.resolve(),
        args.gpu_lock.resolve(),
        preflight=args.command == "preflight",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
