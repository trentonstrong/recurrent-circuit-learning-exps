#!/usr/bin/env python3
"""R002 validation and execution entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Must be set before importing JAX.
os.environ.setdefault("XLA_FLAGS", " --xla_gpu_deterministic_ops=true")

from recurrent_circuit_learning.r002 import CONDITIONS, validate_config_contract
from recurrent_circuit_learning.r002_runner import (
    preflight_condition,
    run_condition,
    validate_common_kernel,
    validate_fp64_gradients,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-kernel")
    validate_parser.add_argument("--report", type=Path, required=True)

    fp64_parser = subparsers.add_parser("validate-fp64")
    fp64_parser.add_argument("--report", type=Path, required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--condition", choices=tuple(CONDITIONS), required=True
    )
    preflight_parser.add_argument("--config", type=Path, required=True)
    preflight_parser.add_argument("--report", type=Path, required=True)
    preflight_parser.add_argument("--checkpoint", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--results-root", type=Path, default=Path("results"))
    run_parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    run_parser.add_argument(
        "--gpu-lock", type=Path, default=Path("envs/jax-gpu/uv.lock")
    )

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--config", type=Path, required=True)
    sweep_parser.add_argument("--concurrency", type=int, required=True)
    sweep_parser.add_argument("--sweep-id", required=True)
    sweep_parser.add_argument("--results-root", type=Path, default=Path("results"))
    sweep_parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    sweep_parser.add_argument(
        "--gpu-lock", type=Path, default=Path("envs/jax-gpu/uv.lock")
    )

    args = parser.parse_args()
    if args.command == "validate-kernel":
        report = validate_common_kernel()
        write_json(args.report, report)
    elif args.command == "validate-fp64":
        report = validate_fp64_gradients()
        write_json(args.report, report)
    elif args.command == "preflight":
        config_bytes = args.config.read_bytes()
        document = json.loads(config_bytes)
        validate_config_contract(document)
        report = preflight_condition(
            CONDITIONS[args.condition],
            args.checkpoint,
            hashlib.sha256(config_bytes).hexdigest(),
        )
        write_json(args.report, report)
    elif args.command == "run":
        report = run_condition(
            args.config,
            args.seed,
            CONDITIONS[args.condition],
            args.results_root / args.run_id,
            args.artifacts_root / args.run_id,
            args.gpu_lock,
        )
    else:
        if args.concurrency != 1:
            raise ValueError("validated R002 launch currently requires --concurrency 1")
        config_bytes = args.config.read_bytes()
        document = json.loads(config_bytes)
        seeds, conditions = validate_config_contract(document)
        completed = []
        for seed in seeds:
            for condition in conditions.values():
                run_id = f"R002_{args.sweep_id}_seed{seed:02d}_{condition.name}"
                result = run_condition(
                    args.config,
                    seed,
                    condition,
                    args.results_root / run_id,
                    args.artifacts_root / run_id,
                    args.gpu_lock,
                )
                completed.append({"run_id": run_id, "status": result["status"]})
        report = {"status": "passed", "runs": completed}
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
