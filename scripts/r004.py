#!/usr/bin/env python3
"""R004 preflight, continuation, sweep, and offline analysis entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# These settings are part of the R002/R004 executable random and GPU contract.
os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault("XLA_FLAGS", " --xla_gpu_deterministic_ops=true")

import numpy as np

from recurrent_circuit_learning.r002 import CONDITIONS
from recurrent_circuit_learning.r000 import sha256_file
from recurrent_circuit_learning.r004_analysis import run_analysis
from recurrent_circuit_learning.r004_runner import (
    load_parent,
    run_condition,
    run_preflight,
    validate_config_contract,
    write_json,
)


def _passed_preflight(results_root: Path, config_sha256: str) -> Path:
    matches = []
    for path in sorted(results_root.glob("R004_preflight_*/validation.json")):
        report = json.loads(path.read_text())
        if (
            report.get("status") == "passed"
            and report.get("config_sha256") == config_sha256
            and all(
                (Path(__file__).resolve().parents[1] / source).is_file()
                and sha256_file(Path(__file__).resolve().parents[1] / source)
                == identity["sha256"]
                for source, identity in report.get("code", {})
                .get("implementation_sources", {})
                .items()
            )
            and len(report.get("code", {}).get("implementation_sources", {})) == 3
        ):
            matches.append(path)
    if not matches:
        raise ValueError("a passed matching R004 preflight is required before launch")
    return matches[-1]


def _validate_parent_cohort(
    repository_root: Path,
    artifact_root: Path,
    seeds: tuple[int, ...],
    conditions: dict[str, object],
) -> None:
    for seed in seeds:
        parents = [
            load_parent(repository_root, artifact_root, seed, condition)
            for condition in conditions.values()
        ]
        if (
            len({tuple(np.asarray(parent.data_key).tolist()) for parent in parents})
            != 1
        ):
            raise ValueError(f"seed {seed} parent data keys are not paired")
        if (
            len({parent.manifest["pairing"]["wiring_sha256"] for parent in parents})
            != 1
        ):
            raise ValueError(f"seed {seed} parent wiring is not paired")
        if (
            len(
                {
                    parent.manifest["pairing"]["training_stream_sha256"]
                    for parent in parents
                }
            )
            != 1
        ):
            raise ValueError(f"seed {seed} parent training streams are not paired")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--config", type=Path, required=True)
    preflight_parser.add_argument("--run-id", required=True)
    preflight_parser.add_argument("--results-root", type=Path, default=Path("results"))
    preflight_parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts")
    )
    preflight_parser.add_argument(
        "--parent-artifacts-root", type=Path, default=Path("artifacts")
    )
    preflight_parser.add_argument(
        "--gpu-lock", type=Path, default=Path("envs/jax-gpu/uv.lock")
    )

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--results-root", type=Path, default=Path("results"))
    run_parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    run_parser.add_argument(
        "--parent-artifacts-root", type=Path, default=Path("artifacts")
    )
    run_parser.add_argument(
        "--gpu-lock", type=Path, default=Path("envs/jax-gpu/uv.lock")
    )

    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--config", type=Path, required=True)
    sweep_parser.add_argument("--concurrency", type=int, required=True)
    sweep_parser.add_argument("--sweep-id", required=True)
    sweep_parser.add_argument("--resume", action="store_true")
    sweep_parser.add_argument("--results-root", type=Path, default=Path("results"))
    sweep_parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    sweep_parser.add_argument(
        "--parent-artifacts-root", type=Path, default=Path("artifacts")
    )
    sweep_parser.add_argument(
        "--gpu-lock", type=Path, default=Path("envs/jax-gpu/uv.lock")
    )

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--config", type=Path, required=True)
    analyze_parser.add_argument("--sweep-id", required=True)
    analyze_parser.add_argument("--analysis-id", required=True)
    analyze_parser.add_argument("--results-root", type=Path, default=Path("results"))
    analyze_parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts")
    )

    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    document = json.loads(config_bytes)
    seeds, conditions = validate_config_contract(document)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    repository_root = args.config.resolve().parents[2]

    if args.command == "preflight":
        output_id = f"R004_{args.run_id}"
        report = run_preflight(
            args.config,
            args.run_id,
            args.results_root / output_id,
            args.artifacts_root / output_id,
            args.gpu_lock,
            args.parent_artifacts_root,
        )
    elif args.command == "run":
        _passed_preflight(args.results_root, config_sha256)
        report = run_condition(
            args.config,
            args.seed,
            CONDITIONS[args.condition],
            args.results_root / args.run_id,
            args.artifacts_root / args.run_id,
            args.gpu_lock,
            parent_artifact_root=args.parent_artifacts_root,
            resume=args.resume,
        )
    elif args.command == "sweep":
        if args.concurrency != 1:
            raise ValueError("R004's validated GPU launch requires --concurrency 1")
        preflight_path = _passed_preflight(args.results_root, config_sha256)
        _validate_parent_cohort(
            repository_root, args.parent_artifacts_root, seeds, conditions
        )
        aggregate_directory = args.results_root / f"R004_{args.sweep_id}"
        if aggregate_directory.exists() and not args.resume:
            raise FileExistsError("R004 aggregate sweep directory already exists")
        aggregate_directory.mkdir(parents=True, exist_ok=True)
        failures = []
        completed = []
        for seed in seeds:
            for condition in conditions.values():
                run_id = f"R004_{args.sweep_id}_seed{seed:02d}_{condition.name}"
                print(f"START {run_id}", flush=True)
                try:
                    result = run_condition(
                        args.config,
                        seed,
                        condition,
                        args.results_root / run_id,
                        args.artifacts_root / run_id,
                        args.gpu_lock,
                        parent_artifact_root=args.parent_artifacts_root,
                        resume=args.resume,
                    )
                    completed.append({"run_id": run_id, "status": result["status"]})
                    print(f"COMPLETE {run_id}", flush=True)
                except BaseException as error:
                    failure = {
                        "run_id": run_id,
                        "seed": seed,
                        "condition": condition.name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    failures.append(failure)
                    write_json(aggregate_directory / "failure_ledger.json", failures)
                    print(f"FAILED {run_id}: {failure['error']}", flush=True)
        continuation_pairing = {}
        if not failures and len(completed) == 64:
            for seed in seeds:
                seed_manifests = [
                    json.loads(
                        (
                            args.results_root
                            / f"R004_{args.sweep_id}_seed{seed:02d}_{condition.name}"
                            / "manifest.json"
                        ).read_text()
                    )
                    for condition in conditions.values()
                ]
                digests = [
                    manifest["continuation_batch_hashes_sha256"]
                    for manifest in seed_manifests
                ]
                continuation_pairing[str(seed)] = {
                    "digests": dict(zip(conditions, digests)),
                    "all_conditions_equal": len(set(digests)) == 1,
                }
            if not all(
                item["all_conditions_equal"] for item in continuation_pairing.values()
            ):
                failures.append(
                    {
                        "run_id": None,
                        "seed": None,
                        "condition": None,
                        "error": "continuation batch-stream pairing mismatch",
                    }
                )
        report = {
            "schema_version": 1,
            "experiment_id": "R004",
            "sweep_id": args.sweep_id,
            "status": "completed"
            if not failures and len(completed) == 64
            else "failed",
            "config_sha256": config_sha256,
            "preflight": str(preflight_path),
            "scheduled_runs": 64,
            "completed_runs": len(completed),
            "runs": completed,
            "continuation_batch_pairing": continuation_pairing,
            "failures": failures,
        }
        write_json(aggregate_directory / "manifest.json", report)
        write_json(aggregate_directory / "failure_ledger.json", failures)
    else:
        analysis_output_id = f"R004_{args.sweep_id}_{args.analysis_id}"
        report = run_analysis(
            args.config,
            args.sweep_id,
            args.analysis_id,
            args.results_root / analysis_output_id,
            args.artifacts_root / analysis_output_id,
            args.artifacts_root,
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] not in {"passed", "completed"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
