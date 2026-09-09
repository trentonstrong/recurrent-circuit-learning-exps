"""Canonical R001 synchronous checkerboard experiment runner."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import save_checkpoint
from .difflogic_ca import (
    batch_rollout,
    batch_trajectory,
    hard_gate_ids,
    init_train_state,
    make_optimizer,
    make_target,
    make_train_step,
    sample_training_batch,
    validate_config_contract,
)
from .r000 import environment_metadata, sha256_file


def _command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _artifact(path: Path, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _save_probe_set(path: Path, inputs: jax.Array, target: jax.Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, inputs=np.asarray(inputs), target=np.asarray(target))


def _save_hard_circuit(path: Path, params, wires) -> None:
    arrays: dict[str, np.ndarray] = {}
    gates = hard_gate_ids(params)
    for network in ("perceive", "update"):
        for index, values in enumerate(gates[network]):
            arrays[f"gate_{network}_{index:02d}"] = np.asarray(values)
        for index, (wire_a, wire_b) in enumerate(wires[network]):
            arrays[f"wire_{network}_{index:02d}_a"] = np.asarray(wire_a)
            arrays[f"wire_{network}_{index:02d}_b"] = np.asarray(wire_b)
    np.savez_compressed(path, **arrays)


def _git_metadata() -> dict[str, Any]:
    status = _command("git", "status", "--porcelain=v1")
    metadata: dict[str, Any] = {
        "commit": _command("git", "rev-parse", "HEAD"),
        "dirty": bool(status),
    }
    if status:
        patch = subprocess.check_output(["git", "diff", "--binary"])
        metadata["tracked_patch_sha256"] = hashlib.sha256(patch).hexdigest()
        metadata["status"] = status.splitlines()
    return metadata


def _hardware_metadata() -> dict[str, Any]:
    gpu_line = _command(
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader",
    )
    return {
        "kernel": platform.release(),
        "platform": platform.platform(),
        "gpu_query": gpu_line,
        "cpu": _command(
            "bash", "-lc", "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"
        ),
    }


def _gradient_metrics(gradients) -> tuple[float, float]:
    leaves = jax.tree_util.tree_leaves(gradients)
    max_abs = jnp.max(jnp.stack([jnp.max(jnp.abs(value)) for value in leaves]))
    squared = sum(jnp.sum(jnp.square(value)) for value in leaves)
    return float(max_abs), float(jnp.sqrt(squared))


def _evaluation_metrics(
    soft: np.ndarray,
    hard: np.ndarray,
    target: jax.Array,
) -> dict[str, Any]:
    target_array = np.asarray(target)
    soft_loss = float(np.square(soft[..., 0] - target_array[..., 0]).sum())
    hard_errors_by_grid = np.count_nonzero(
        hard[..., 0] != target_array[..., 0], axis=(1, 2)
    )
    hard_bit_errors = int(hard_errors_by_grid.sum())
    hard_loss = float(np.square(hard[..., 0] - target_array[..., 0]).sum())
    metrics = {
        "soft_terminal_summed_squared_error": soft_loss,
        "hard_terminal_summed_squared_error": hard_loss,
        "soft_hard_gap": hard_loss - soft_loss,
        "hard_bit_errors": hard_bit_errors,
        "perfect_grid_count": int(np.count_nonzero(hard_errors_by_grid == 0)),
        "perfect_grid_fraction": float(np.mean(hard_errors_by_grid == 0)),
        "exact_all_32": hard_bit_errors == 0,
    }
    return metrics


def run(
    config_path: Path,
    result_directory: Path,
    artifact_directory: Path,
    gpu_lock_path: Path,
) -> dict[str, Any]:
    if result_directory.exists() or artifact_directory.exists():
        raise FileExistsError("run result or artifact directory already exists")
    result_directory.mkdir(parents=True)
    artifact_directory.mkdir(parents=True)

    config_bytes = config_path.read_bytes()
    document = json.loads(config_bytes)
    config = validate_config_contract(document)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"R001 requires the GPU runner, got {jax.default_backend()}")
    if jax.config.jax_threefry_partitionable:
        raise RuntimeError(
            "legacy source RNG requires jax_threefry_partitionable=false"
        )

    started_at = _utc_now()
    optimizer = make_optimizer(config)
    train_step = make_train_step(config, optimizer)
    state, wires = init_train_state(config, optimizer)
    data_key = jax.random.PRNGKey(config.seed)
    target = make_target(config)

    evaluation_config = dataclasses.replace(config, batch_size=32)
    _, evaluation_inputs = sample_training_batch(
        jax.random.PRNGKey(document["project_additions"]["evaluation"]["seed"]),
        evaluation_config,
    )
    evaluation_target = make_target(evaluation_config)
    probe_path = artifact_directory / "fixed_evaluation_set.npz"
    _save_probe_set(probe_path, evaluation_inputs, evaluation_target)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "R001",
        "run_id": result_directory.name,
        "status": "running",
        "started_at": started_at,
        "ended_at": None,
        "current_update": 0,
        "primary_checkpoint_update": 500,
        "config": document,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_sha256": document["source_sha256"],
        "code": _git_metadata(),
        "environment": {
            **environment_metadata(),
            "uv": _command("uv", "--version"),
            "lock_path": str(gpu_lock_path),
            "lock_sha256": sha256_file(gpu_lock_path),
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "allocator": {
                name: os.environ.get(name)
                for name in (
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION",
                )
            },
        },
        "hardware": _hardware_metadata(),
        "rng": {
            "implementation": "JAX threefry2x32 with jax_threefry_partitionable=false",
            "training_seed": config.seed,
            "probe_seed": document["project_additions"]["evaluation"]["seed"],
            "training_data_and_model_streams": "independent source-compatible split streams",
        },
        "metric_definitions": {
            "training_loss": "pre-update terminal channel-0 squared error summed over batch and grid",
            "probe_soft_loss": "terminal channel-0 squared error summed over 32 fixed grids",
            "hard_bit_error": "terminal native-argmax channel-0 mismatches on 32 fixed grids",
            "exact": "zero hard bit errors across all 32 fixed grids",
        },
        "artifacts": [_artifact(probe_path, "fixed_evaluation_set")],
        "checkpoints": [],
        "evaluations": [],
        "failure_reason": None,
    }
    manifest_path = result_directory / "manifest.json"
    metrics_path = result_directory / "metrics.jsonl"
    _write_json(manifest_path, manifest)

    try:
        # Lower/compile against a copied sample. Neither canonical RNG stream advances.
        _, compile_inputs = sample_training_batch(data_key, config)
        compile_start = time.perf_counter()
        compiled_step = train_step.lower(state, compile_inputs, target, wires).compile()
        manifest["training_compile_seconds"] = time.perf_counter() - compile_start

        evaluation_function = jax.jit(
            lambda current: (
                batch_rollout(
                    evaluation_inputs,
                    current,
                    wires,
                    True,
                    config.periodic,
                    config.runtime_ticks,
                ),
                batch_rollout(
                    evaluation_inputs,
                    current,
                    wires,
                    False,
                    config.periodic,
                    config.runtime_ticks,
                ),
            )
        )
        trajectory_function = jax.jit(
            lambda current: (
                batch_trajectory(
                    evaluation_inputs[:1],
                    current,
                    wires,
                    True,
                    config.periodic,
                    config.runtime_ticks,
                ),
                batch_trajectory(
                    evaluation_inputs[:1],
                    current,
                    wires,
                    False,
                    config.periodic,
                    config.runtime_ticks,
                ),
            )
        )
        evaluation_compile_start = time.perf_counter()
        compiled_evaluation = evaluation_function.lower(state.params).compile()
        compiled_trajectory = trajectory_function.lower(state.params).compile()
        manifest["evaluation_compile_seconds"] = (
            time.perf_counter() - evaluation_compile_start
        )

        checkpoint_metadata = {
            "run_id": result_directory.name,
            "config_sha256": manifest["config_sha256"],
            "code_commit": manifest["code"]["commit"],
            "environment_lock_sha256": manifest["environment"]["lock_sha256"],
        }

        def save_evaluation(update: int) -> None:
            soft, hard = jax.device_get(compiled_evaluation(state.params))
            evaluation = _evaluation_metrics(
                np.asarray(soft), np.asarray(hard), evaluation_target
            )
            trajectory, hard_trajectory = jax.device_get(
                compiled_trajectory(state.params)
            )
            trajectory_path = (
                artifact_directory / f"probe_trajectory_update_{update:03d}.npz"
            )
            np.savez_compressed(
                trajectory_path,
                soft=np.asarray(trajectory),
                hard=np.asarray(hard_trajectory),
            )
            checkpoint_path = artifact_directory / f"checkpoint_update_{update:03d}.npz"
            save_checkpoint(checkpoint_path, state, data_key, checkpoint_metadata)
            evaluation["update"] = update
            manifest["evaluations"].append(evaluation)
            manifest["checkpoints"].append(
                {
                    "update": update,
                    "checkpoint": _artifact(checkpoint_path, "training_checkpoint"),
                    "probe_trajectory": _artifact(
                        trajectory_path, "selected_probe_trajectory"
                    ),
                }
            )
            _write_json(manifest_path, manifest)

        save_evaluation(0)
        training_started = time.perf_counter()
        update_durations: list[float] = []
        with metrics_path.open("w") as metrics_handle:
            for update in range(1, config.optimizer_updates + 1):
                data_key, inputs = sample_training_batch(data_key, config)
                update_started = time.perf_counter()
                state, soft_loss, auxiliary, gradients = compiled_step(
                    state, inputs, target, wires
                )
                jax.block_until_ready(state.params)
                duration = time.perf_counter() - update_started
                update_durations.append(duration)
                gradient_max, gradient_l2 = _gradient_metrics(gradients)
                metric = {
                    "update": update,
                    "pre_update_soft_loss": float(soft_loss),
                    "pre_update_hard_loss": float(auxiliary["hard"]),
                    "gradient_max_abs": gradient_max,
                    "gradient_l2": gradient_l2,
                    "synchronized_seconds": duration,
                }
                metrics_handle.write(json.dumps(metric, sort_keys=True) + "\n")
                metrics_handle.flush()
                manifest["current_update"] = update
                if update % 50 == 0:
                    save_evaluation(update)

        manifest["training_synchronized_seconds"] = (
            time.perf_counter() - training_started
        )
        manifest["median_update_seconds"] = float(np.median(update_durations))
        manifest["final_training_metric"] = json.loads(
            metrics_path.read_text().splitlines()[-1]
        )
        final_circuit_path = artifact_directory / "final_hard_circuit.npz"
        _save_hard_circuit(final_circuit_path, state.params, wires)
        manifest["artifacts"].append(
            _artifact(final_circuit_path, "final_hard_circuit")
        )
        memory_stats = jax.devices()[0].memory_stats() or {}
        manifest["device_memory"] = {
            key: int(value)
            for key, value in memory_stats.items()
            if key in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"}
            and isinstance(value, (int, np.integer))
        }
        manifest["status"] = "completed"
        manifest["ended_at"] = _utc_now()
        _write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["ended_at"] = _utc_now()
        manifest["failure_reason"] = f"{type(error).__name__}: {error}"
        _write_json(manifest_path, manifest)
        raise
    return manifest
