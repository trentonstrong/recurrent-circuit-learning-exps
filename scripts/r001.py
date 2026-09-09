#!/usr/bin/env python3
"""Run the canonical R001 experiment in a fresh result identity."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# This must be set before JAX initializes XLA. It is accepted by the validated
# modern runtime and is the flag retained from the pinned notebook.
os.environ.setdefault("XLA_FLAGS", " --xla_gpu_deterministic_ops=true")

from recurrent_circuit_learning.r001 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/r001_sync_reference.json"),
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--gpu-lock", type=Path, default=Path("envs/jax-gpu/uv.lock"))
    args = parser.parse_args()
    manifest = run(
        args.config,
        args.results_root / args.run_id,
        args.artifacts_root / args.run_id,
        args.gpu_lock,
    )
    print(
        f"{manifest['run_id']}: {manifest['status']} at update "
        f"{manifest['current_update']}"
    )


if __name__ == "__main__":
    main()
