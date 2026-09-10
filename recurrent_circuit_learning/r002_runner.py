"""Validation, preflight, and formal execution support for R002."""

from __future__ import annotations

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
import optax

from .difflogic_ca import (
    batch_rollout as source_batch_rollout,
)
from .difflogic_ca import (
    bin_op_s,
    decode_soft,
    make_target,
    nontrivial_fixed_params,
    sample_training_batch,
)
from .r000 import environment_metadata, sha256_file
from .r002 import (
    CONDITIONS,
    GATE_TRUTH_TABLES,
    Condition,
    batch_trajectory_q,
    config_for,
    effective_truth_tables,
    evaluate_params,
    gate_ids,
    hardened_truth_tables,
    init_condition_state,
    init_paired,
    load_checkpoint,
    loss,
    make_optimizer,
    make_train_step,
    multilinear_gate,
    ordered_training_stream,
    q_displacement,
    save_checkpoint,
    stream_sha256,
    tree_sha256,
    validate_config_contract,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _comparison(actual: Any, expected: Any, atol: float = 1e-6, rtol: float = 1e-4):
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    difference = np.abs(
        actual_array.astype(np.float64) - expected_array.astype(np.float64)
    )
    denominator = np.maximum(
        np.abs(expected_array.astype(np.float64)), np.finfo(np.float64).tiny
    )
    return {
        "ok": bool(np.allclose(actual_array, expected_array, atol=atol, rtol=rtol)),
        "max_abs_error": float(difference.max(initial=0.0)),
        "max_relative_error": float((difference / denominator).max(initial=0.0)),
        "atol": atol,
        "rtol": rtol,
    }


def _flatten(tree: Any) -> np.ndarray:
    return np.concatenate(
        [np.asarray(value).reshape(-1) for value in jax.tree_util.tree_leaves(tree)]
    )


def _source_params_for_q(q_params):
    return jax.tree_util.tree_map(
        lambda q: jax.nn.one_hot(
            jnp.sum(
                (q >= 0.5).astype(jnp.int32) * jnp.array([8, 4, 2, 1], dtype=jnp.int32),
                axis=-1,
            ),
            16,
            dtype=jnp.float32,
        ),
        q_params,
    )


def _optimizer_step(params, gradients, optimizer):
    state = optimizer.init(params)
    updates, state = optimizer.update(gradients, state, params)
    return optax.apply_updates(params, updates), state


def validate_common_kernel(seed: int = 23) -> dict[str, Any]:
    """Validate the q bridge, matched functions, derivatives, and exports on GPU."""
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"R002 runtime validation requires GPU, got {jax.default_backend()}"
        )
    config = config_for(seed, CONDITIONS["categorical_reference_decay"])
    paired = init_paired(seed)
    _, batches = ordered_training_stream(seed, config, 1)
    inputs = batches[0]
    target = make_target(config)
    initial_q = effective_truth_tables(paired.categorical_params, "categorical")
    truth_q = effective_truth_tables(paired.truth_params, "truth")
    initial_match = _comparison(_flatten(truth_q), _flatten(initial_q), 1e-7, 1e-6)

    gate_logits = jnp.sin(jnp.arange(16 * 7, dtype=jnp.float32).reshape(7, 16) * 0.137)
    gate_q = jnp.matmul(
        decode_soft(gate_logits),
        GATE_TRUTH_TABLES,
        precision=jax.lax.Precision.HIGHEST,
    )
    fractions = jnp.linspace(0, 1, 7, dtype=jnp.float32)
    gate_source = bin_op_s(fractions, fractions[::-1], decode_soft(gate_logits))
    gate_common = multilinear_gate(fractions, fractions[::-1], gate_q)

    cases: dict[str, Any] = {}
    for label, categorical in (
        ("initial", paired.categorical_params),
        ("nontrivial", nontrivial_fixed_params(paired.categorical_params)),
    ):
        truth = jax.tree_util.tree_map(
            lambda q: jnp.log(q) - jnp.log1p(-q),
            effective_truth_tables(categorical, "categorical"),
        )
        source_trajectory = jax.jit(
            lambda p: _source_trajectory(inputs, p, paired.wires, config)
        )(categorical)
        categorical_trajectory = jax.jit(
            lambda p: batch_trajectory_q(
                inputs,
                effective_truth_tables(p, "categorical"),
                paired.wires,
                config.periodic,
                config.runtime_ticks,
            )
        )(categorical)
        truth_trajectory = jax.jit(
            lambda p: batch_trajectory_q(
                inputs,
                effective_truth_tables(p, "truth"),
                paired.wires,
                config.periodic,
                config.runtime_ticks,
            )
        )(truth)
        source_loss, source_grad = jax.jit(
            jax.value_and_grad(
                lambda p: jnp.square(
                    source_batch_rollout(
                        inputs,
                        p,
                        paired.wires,
                        True,
                        config.periodic,
                        config.runtime_ticks,
                    )[..., 0]
                    - target[..., 0]
                ).sum()
            )
        )(categorical)
        common_loss, common_grad = jax.jit(
            jax.value_and_grad(
                lambda p: loss(p, paired.wires, inputs, target, "categorical", config)
            )
        )(categorical)
        tangent = jnp.cos(
            jnp.arange(inputs.size, dtype=jnp.float32).reshape(inputs.shape) * 0.01
        )
        _, categorical_jvp = jax.jvp(
            lambda value, current=categorical: evaluate_params(
                value, current, paired.wires, "categorical", config
            ),
            (inputs,),
            (tangent,),
        )
        _, truth_jvp = jax.jvp(
            lambda value, current=truth: evaluate_params(
                value, current, paired.wires, "truth", config
            ),
            (inputs,),
            (tangent,),
        )
        source_optimizer = make_optimizer(config)
        source_updated, source_opt_state = _optimizer_step(
            categorical, source_grad, source_optimizer
        )
        common_updated, common_opt_state = _optimizer_step(
            categorical, common_grad, make_optimizer(config)
        )
        cases[label] = {
            "source_to_common_trajectory": _comparison(
                categorical_trajectory, source_trajectory
            ),
            "matched_coordinate_trajectory": _comparison(
                truth_trajectory, categorical_trajectory
            ),
            "source_to_common_loss": _comparison(common_loss, source_loss),
            "source_to_common_gradient": _comparison(
                _flatten(common_grad), _flatten(source_grad)
            ),
            "source_to_common_updated_params": _comparison(
                _flatten(common_updated), _flatten(source_updated), 5e-5, 1e-4
            ),
            "source_to_common_optimizer_state": _comparison(
                _flatten(common_opt_state), _flatten(source_opt_state)
            ),
            "matched_state_jvp": _comparison(truth_jvp, categorical_jvp),
        }

    exports: dict[str, Any] = {}
    for representation, params in (
        ("categorical", nontrivial_fixed_params(paired.categorical_params)),
        (
            "truth",
            jax.tree_util.tree_map(
                lambda q: jnp.log(q) - jnp.log1p(-q),
                effective_truth_tables(
                    nontrivial_fixed_params(paired.categorical_params), "categorical"
                ),
            ),
        ),
    ):
        for hardening in ("native", "common"):
            q = hardened_truth_tables(params, representation, hardening)
            source_params = _source_params_for_q(q)
            common_hard = evaluate_params(
                inputs, params, paired.wires, representation, config, hardening
            )
            source_hard = source_batch_rollout(
                inputs,
                source_params,
                paired.wires,
                False,
                config.periodic,
                config.runtime_ticks,
            )
            exports[f"{representation}_{hardening}"] = {
                "execution": _comparison(common_hard, source_hard, 0.0, 0.0),
                "gate_id_sha256": tree_sha256(
                    gate_ids(params, representation, hardening)
                ),
            }

    comparisons = [initial_match, _comparison(gate_common, gate_source)]
    comparisons.extend(item for case in cases.values() for item in case.values())
    comparisons.extend(item["execution"] for item in exports.values())
    return {
        "schema_version": 1,
        "status": "passed" if all(item["ok"] for item in comparisons) else "failed",
        "seed": seed,
        "environment": environment_metadata(),
        "initial_effective_table_match": initial_match,
        "gate_source_to_common": _comparison(gate_common, gate_source),
        "cases": cases,
        "exports": exports,
        "parameter_gradient_note": "Gradients across coordinate systems are intentionally not compared.",
    }


def _source_trajectory(inputs, params, wires, config):
    def one_grid(grid):
        def body(current, _unused):
            from .difflogic_ca import run_sync

            updated = run_sync(current, params, wires, True, config.periodic)
            return updated, updated

        _, states = jax.lax.scan(body, grid, None, length=config.runtime_ticks)
        return jnp.concatenate([grid[None, ...], states], axis=0)

    return jnp.swapaxes(jax.vmap(one_grid)(inputs), 0, 1)


def validate_fp64_gradients() -> dict[str, Any]:
    if not jax.config.x64_enabled:
        raise RuntimeError("FP64 validation requires JAX_ENABLE_X64=true")
    a = jnp.array([0.13, 0.41, 0.79], dtype=jnp.float64)
    b = jnp.array([0.88, 0.37, 0.21], dtype=jnp.float64)
    categorical = jnp.cos(jnp.arange(16, dtype=jnp.float64) * 0.23)
    q = jnp.matmul(
        jax.nn.softmax(categorical),
        GATE_TRUTH_TABLES.astype(jnp.float64),
        precision=jax.lax.Precision.HIGHEST,
    )
    truth = jnp.log(q) - jnp.log1p(-q)
    target = jnp.array([0.2, 0.7, 0.4], dtype=jnp.float64)

    def objectives():
        return {
            "categorical": lambda value: jnp.square(
                multilinear_gate(
                    a,
                    b,
                    jnp.matmul(
                        jax.nn.softmax(value),
                        GATE_TRUTH_TABLES.astype(jnp.float64),
                        precision=jax.lax.Precision.HIGHEST,
                    ),
                )
                - target
            ).sum(),
            "truth": lambda value: jnp.square(
                multilinear_gate(a, b, jax.nn.sigmoid(value)) - target
            ).sum(),
        }

    reports = {}
    epsilon = 1e-5
    for name, function in objectives().items():
        value = categorical if name == "categorical" else truth
        automatic = jax.grad(function)(value)
        finite = np.empty(value.shape, dtype=np.float64)
        for index in range(value.size):
            direction = jnp.zeros_like(value).at[index].set(epsilon)
            finite[index] = float(
                (function(value + direction) - function(value - direction))
                / (2 * epsilon)
            )
        reports[name] = _comparison(automatic, finite, 1e-8, 1e-6)
    return {
        "schema_version": 1,
        "status": "passed"
        if all(item["ok"] for item in reports.values())
        else "failed",
        "environment": environment_metadata(),
        "central_difference_epsilon": epsilon,
        "representations": reports,
    }


def _evaluation_metrics(predicted: np.ndarray, target: jax.Array) -> dict[str, Any]:
    target_array = np.asarray(target)
    errors = np.count_nonzero(predicted[..., 0] != target_array[..., 0])
    return {
        "summed_squared_error": float(
            np.square(predicted[..., 0] - target_array[..., 0]).sum()
        ),
        "bit_errors": int(errors),
    }


def preflight_condition(
    condition: Condition,
    checkpoint_path: Path,
    config_sha256: str,
    seed: int = 23,
) -> dict[str, Any]:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"R002 preflight requires GPU, got {jax.default_backend()}")
    config = config_for(seed, condition, optimizer_updates=2)
    optimizer = make_optimizer(config)
    state, wires, paired = init_condition_state(seed, condition, optimizer)
    initial_q = effective_truth_tables(paired.categorical_params, "categorical")
    data_root = jax.random.PRNGKey(seed)
    first_data_key, first_inputs = sample_training_batch(data_root, config)
    final_data_key, second_inputs = sample_training_batch(first_data_key, config)
    batches = [first_inputs, second_inputs]
    target = make_target(config)
    train_step = make_train_step(config, condition.representation, optimizer)
    compile_started = time.perf_counter()
    compiled = train_step.lower(state, batches[0], target, wires).compile()
    compile_seconds = time.perf_counter() - compile_started

    initial_state_hash = tree_sha256(
        (state.params, state.opt_state, state.key, state.update_index)
    )
    durations = []
    first_started = time.perf_counter()
    first, first_loss, first_gradients, first_updates = compiled(
        state, batches[0], target, wires
    )
    jax.block_until_ready(first.params)
    durations.append(time.perf_counter() - first_started)
    checkpoint_metadata = {
        "experiment_id": "R002",
        "seed": seed,
        "condition": condition.name,
        "representation": condition.representation,
        "kernel": "common_multilinear_truth_table",
        "config_sha256": config_sha256,
        "wiring_sha256": tree_sha256(wires),
        "training_stream_sha256": stream_sha256(batches),
    }
    save_checkpoint(checkpoint_path, first, first_data_key, checkpoint_metadata)

    second_started = time.perf_counter()
    uninterrupted, second_loss, _, _ = compiled(first, batches[1], target, wires)
    jax.block_until_ready(uninterrupted.params)
    durations.append(time.perf_counter() - second_started)
    restored, restored_data_key, restored_metadata = load_checkpoint(
        checkpoint_path, state, checkpoint_metadata
    )
    resumed_data_key, resumed_inputs = sample_training_batch(restored_data_key, config)
    resumed, resumed_loss, _, _ = compiled(restored, resumed_inputs, target, wires)
    jax.block_until_ready(resumed.params)
    resume_checks = {
        "metadata": restored_metadata == checkpoint_metadata,
        "data_key": np.array_equal(
            np.asarray(restored_data_key), np.asarray(first_data_key)
        ),
        "next_input": np.array_equal(
            np.asarray(resumed_inputs), np.asarray(batches[1])
        ),
        "next_data_key": np.array_equal(
            np.asarray(resumed_data_key), np.asarray(final_data_key)
        ),
        "loss": np.array_equal(np.asarray(resumed_loss), np.asarray(second_loss)),
        "params": tree_sha256(resumed.params) == tree_sha256(uninterrupted.params),
        "optimizer": tree_sha256(resumed.opt_state)
        == tree_sha256(uninterrupted.opt_state),
        "model_key": np.array_equal(
            np.asarray(resumed.key), np.asarray(uninterrupted.key)
        ),
        "update_index": int(resumed.update_index)
        == int(uninterrupted.update_index)
        == 2,
    }
    final_evaluations = {}
    for hardening in ("native", "common"):
        prediction = jax.jit(
            lambda params, current_hardening=hardening: evaluate_params(
                batches[1],
                params,
                wires,
                condition.representation,
                config,
                current_hardening,
            )
        )(uninterrupted.params)
        final_evaluations[hardening] = _evaluation_metrics(
            np.asarray(prediction), target
        )
    memory_stats = jax.devices()[0].memory_stats() or {}
    initial_coordinate_q = effective_truth_tables(
        state.params, condition.representation
    )
    report = {
        "schema_version": 1,
        "kind": "R002_development_preflight",
        "status": "passed" if all(resume_checks.values()) else "failed",
        "seed": seed,
        "formal_cohort": False,
        "condition": condition.name,
        "representation": condition.representation,
        "weight_decay": condition.weight_decay,
        "kernel": "common_multilinear_truth_table",
        "optimizer_updates": 2,
        "full_shape": {
            "grid": list(config.grid_size),
            "batch": config.batch_size,
            "ticks": config.runtime_ticks,
        },
        "pairing": {
            "wiring_sha256": tree_sha256(wires),
            "categorical_initial_params_sha256": tree_sha256(paired.categorical_params),
            "truth_initial_params_sha256": tree_sha256(paired.truth_params),
            "training_stream_sha256": stream_sha256(batches),
            "training_stream_final_key": np.asarray(final_data_key).tolist(),
        },
        "initial_effective_table_discrepancy": _comparison(
            _flatten(initial_coordinate_q), _flatten(initial_q), 1e-7, 1e-6
        ),
        "state_reset_after_compilation": tree_sha256(
            (state.params, state.opt_state, state.key, state.update_index)
        )
        == initial_state_hash,
        "compile_seconds": compile_seconds,
        "synchronized_update_seconds": durations,
        "first_pre_update_soft_loss": float(first_loss),
        "second_pre_update_soft_loss": float(second_loss),
        "first_gradient_max_abs": float(
            max(
                jnp.max(jnp.abs(value))
                for value in jax.tree_util.tree_leaves(first_gradients)
            )
        ),
        "first_clipped_element_count": int(
            sum(
                jnp.count_nonzero(jnp.abs(value) > config.clip_value)
                for value in jax.tree_util.tree_leaves(first_gradients)
            )
        ),
        "first_parameter_update_l2": float(
            jnp.sqrt(
                sum(
                    jnp.sum(jnp.square(value))
                    for value in jax.tree_util.tree_leaves(first_updates)
                )
            )
        ),
        "effective_table_displacement_after_two_updates": q_displacement(
            initial_q, uninterrupted.params, condition.representation
        ),
        "resume_checks": resume_checks,
        "evaluation_after_two_updates": final_evaluations,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
        },
        "device_memory": {
            name: int(value)
            for name, value in memory_stats.items()
            if name in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"}
            and isinstance(value, (int, np.integer))
        },
        "environment": environment_metadata(),
        "ended_at": _utc_now(),
    }
    return report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _git_metadata() -> dict[str, Any]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], text=True
    ).strip()
    result: dict[str, Any] = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(status),
    }
    if status:
        result["status"] = status.splitlines()
        result["tracked_patch_sha256"] = hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"])
        ).hexdigest()
    return result


def _artifact(path: Path, kind: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _hard_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    errors_by_grid = np.count_nonzero(predicted[..., 0] != target[..., 0], axis=(1, 2))
    errors = int(errors_by_grid.sum())
    return {
        "terminal_summed_squared_error": float(
            np.square(predicted[..., 0] - target[..., 0]).sum()
        ),
        "bit_errors": errors,
        "perfect_grid_count": int(np.count_nonzero(errors_by_grid == 0)),
        "exact_all_32": errors == 0,
    }


def _readout_cone_masks(wires):
    """Return the one-cell output-0 dependency closure through fixed wiring."""
    update_masks = [None] * len(wires["update"])
    active = {0}
    for index in range(len(wires["update"]) - 1, -1, -1):
        width = wires["update"][index][0].shape[0]
        mask = np.zeros(width, dtype=bool)
        mask[list(active)] = True
        update_masks[index] = mask
        a = np.asarray(wires["update"][index][0])
        b = np.asarray(wires["update"][index][1])
        active = set(a[mask].tolist()) | set(b[mask].tolist())

    active_pairs: set[tuple[int, int]] = set()
    for value in active:
        if value >= 8:
            offset = value - 8
            active_pairs.add((offset % 16, (offset // 16) % 2))
    perceive_masks = [None] * len(wires["perceive"])
    for index in range(len(wires["perceive"]) - 1, -1, -1):
        width = wires["perceive"][index][0].shape[0]
        mask = np.zeros((16, width), dtype=bool)
        for kernel, output in active_pairs:
            mask[kernel, output] = True
        perceive_masks[index] = mask
        a = np.asarray(wires["perceive"][index][0])
        b = np.asarray(wires["perceive"][index][1])
        active_pairs = {
            (kernel, source)
            for kernel, output in active_pairs
            for source in (int(a[output]), int(b[output]))
        }
    return {"perceive": perceive_masks, "update": update_masks}


def _gate_diagnostics(params, wires, representation: str) -> dict[str, Any]:
    q_tree = effective_truth_tables(params, representation)
    masks = _readout_cone_masks(wires)
    layers: dict[str, Any] = {"perceive": [], "update": []}
    for network in ("perceive", "update"):
        for index, q_value in enumerate(q_tree[network]):
            q = np.asarray(q_value, dtype=np.float64)
            clipped = np.clip(
                q, np.finfo(np.float64).tiny, 1 - np.finfo(np.float64).eps
            )
            entropy = -(
                clipped * np.log2(clipped) + (1 - clipped) * np.log2(1 - clipped)
            )
            margin = np.abs(q - 0.5)
            mask = masks[network][index]
            cone_q = q[mask]
            cone_entropy = entropy[mask]
            layers[network].append(
                {
                    "index": index,
                    "gate_count": int(np.prod(q.shape[:-1])),
                    "mean_truth_bit_entropy": float(entropy.mean()),
                    "minimum_rounding_margin": float(margin.min()),
                    "readout_cone_gate_count": int(mask.sum()),
                    "readout_cone_mean_truth_bit_entropy": float(cone_entropy.mean())
                    if cone_entropy.size
                    else None,
                    "readout_cone_minimum_rounding_margin": float(
                        np.abs(cone_q - 0.5).min()
                    )
                    if cone_q.size
                    else None,
                }
            )
    return {
        "definition": (
            "one-cell hard readout dependency closure; recurrent unrolling not expanded"
        ),
        "layers": layers,
    }


def _transition_metrics(before, after, representation: str) -> dict[str, Any]:
    before_q = effective_truth_tables(before, representation)
    after_q = effective_truth_tables(after, representation)
    differences = [
        right - left
        for left, right in zip(
            jax.tree_util.tree_leaves(before_q),
            jax.tree_util.tree_leaves(after_q),
        )
    ]
    result = {
        "q_step_max_abs": float(max(jnp.max(jnp.abs(value)) for value in differences)),
        "q_step_l2": float(
            jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in differences))
        ),
    }
    for hardening in ("native", "common"):
        before_ids = gate_ids(before, representation, hardening)
        after_ids = gate_ids(after, representation, hardening)
        result[f"{hardening}_gate_id_changes"] = int(
            sum(
                jnp.count_nonzero(left != right)
                for left, right in zip(
                    jax.tree_util.tree_leaves(before_ids),
                    jax.tree_util.tree_leaves(after_ids),
                )
            )
        )
    return result


def run_condition(
    config_path: Path,
    seed: int,
    condition: Condition,
    result_directory: Path,
    artifact_directory: Path,
    gpu_lock_path: Path,
) -> dict[str, Any]:
    """Run one fresh formal R002 trial through its fixed 500-update endpoint."""
    if result_directory.exists() or artifact_directory.exists():
        raise FileExistsError("formal R002 result or artifact directory already exists")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"R002 requires the GPU runner, got {jax.default_backend()}")
    if jax.config.jax_threefry_partitionable:
        raise RuntimeError(
            "legacy source RNG requires jax_threefry_partitionable=false"
        )

    config_bytes = config_path.read_bytes()
    document = json.loads(config_bytes)
    seeds, declared = validate_config_contract(document)
    if seed not in seeds or declared[condition.name] != condition:
        raise ValueError("seed or condition is outside the formal R002 cohort")
    config = config_for(seed, condition)
    config_hash = hashlib.sha256(config_bytes).hexdigest()

    probe_path = Path(document["evaluation"]["r001_probe_artifact"])
    expected_probe_hash = document["evaluation"]["r001_probe_sha256"]
    if sha256_file(probe_path) != expected_probe_hash:
        raise ValueError("R001 fixed evaluation artifact hash mismatch")
    with np.load(probe_path, allow_pickle=False) as stored:
        probe_inputs = jnp.asarray(stored["inputs"])
        probe_target = jnp.asarray(stored["target"])

    result_directory.mkdir(parents=True)
    artifact_directory.mkdir(parents=True)
    optimizer = make_optimizer(config)
    state, wires, paired = init_condition_state(seed, condition, optimizer)
    initial_q = effective_truth_tables(paired.categorical_params, "categorical")
    target = make_target(config)
    data_key = jax.random.PRNGKey(seed)
    train_step = make_train_step(config, condition.representation, optimizer)
    _, compile_inputs = sample_training_batch(data_key, config)
    compile_started = time.perf_counter()
    compiled_step = train_step.lower(state, compile_inputs, target, wires).compile()
    compile_seconds = time.perf_counter() - compile_started

    evaluation = jax.jit(
        lambda params: (
            evaluate_params(
                probe_inputs, params, wires, condition.representation, config
            ),
            evaluate_params(
                probe_inputs,
                params,
                wires,
                condition.representation,
                config,
                "native",
            ),
            evaluate_params(
                probe_inputs,
                params,
                wires,
                condition.representation,
                config,
                "common",
            ),
        )
    )
    evaluation_started = time.perf_counter()
    compiled_evaluation = evaluation.lower(state.params).compile()
    q_template = effective_truth_tables(state.params, condition.representation)
    trajectory_function = jax.jit(
        lambda q: batch_trajectory_q(
            probe_inputs[:1], q, wires, config.periodic, config.runtime_ticks
        )
    )
    compiled_trajectory = trajectory_function.lower(q_template).compile()
    evaluation_compile_seconds = time.perf_counter() - evaluation_started

    manifest = {
        "schema_version": 1,
        "experiment_id": "R002",
        "run_id": result_directory.name,
        "status": "running",
        "started_at": _utc_now(),
        "ended_at": None,
        "current_update": 0,
        "seed": seed,
        "condition": condition.name,
        "representation": condition.representation,
        "weight_decay": condition.weight_decay,
        "kernel": "common_multilinear_truth_table",
        "config": document,
        "config_sha256": config_hash,
        "code": _git_metadata(),
        "environment": {
            **environment_metadata(),
            "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
            "lock_path": str(gpu_lock_path),
            "lock_sha256": sha256_file(gpu_lock_path),
            "xla_flags": os.environ.get("XLA_FLAGS"),
        },
        "hardware": {
            "kernel": platform.release(),
            "gpu": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
        },
        "pairing": {
            "wiring_sha256": tree_sha256(wires),
            "categorical_initial_params_sha256": tree_sha256(paired.categorical_params),
            "truth_initial_params_sha256": tree_sha256(paired.truth_params),
            "initial_effective_table_discrepancy": _comparison(
                _flatten(
                    effective_truth_tables(state.params, condition.representation)
                ),
                _flatten(initial_q),
                1e-7,
                1e-6,
            ),
            "training_stream_sha256": None,
        },
        "probe": _artifact(probe_path, "R001_fixed_evaluation_set"),
        "training_compile_seconds": compile_seconds,
        "evaluation_compile_seconds": evaluation_compile_seconds,
        "metric_definitions": {
            "training_loss": (
                "pre-update terminal channel-0 squared error summed over batch and grid"
            ),
            "soft_hard_gap": (
                "hard terminal SSE minus soft terminal SSE on the fixed 32-grid probe"
            ),
            "first_exact": (
                "first saved checkpoint with zero hard bit errors on all 32 probe grids"
            ),
        },
        "evaluations": [],
        "checkpoints": [],
        "first_exact_saved_checkpoint": {"native": None, "common": None},
        "posthoc_boolean_simplification": {
            "status": "pending_completed_hard_circuit",
            "declared_enumeration_cap": 16,
            "note": "analysis only; never changes trial status or training",
        },
        "failure_reason": None,
    }
    manifest_path = result_directory / "manifest.json"
    metrics_path = result_directory / "metrics.jsonl"
    write_json(manifest_path, manifest)
    stream_digest = hashlib.sha256()
    checkpoint_metadata = {
        "experiment_id": "R002",
        "seed": seed,
        "condition": condition.name,
        "representation": condition.representation,
        "kernel": "common_multilinear_truth_table",
        "config_sha256": config_hash,
        "wiring_sha256": manifest["pairing"]["wiring_sha256"],
    }

    def save_evaluation(update: int) -> None:
        soft, native, common = jax.device_get(compiled_evaluation(state.params))
        soft_array = np.asarray(soft)
        target_array = np.asarray(probe_target)
        soft_loss = float(np.square(soft_array[..., 0] - target_array[..., 0]).sum())
        native_metrics = _hard_metrics(np.asarray(native), target_array)
        common_metrics = _hard_metrics(np.asarray(common), target_array)
        record = {
            "update": update,
            "soft_terminal_summed_squared_error": soft_loss,
            "native": native_metrics,
            "common": common_metrics,
            "native_soft_hard_gap": (
                native_metrics["terminal_summed_squared_error"] - soft_loss
            ),
            "common_soft_hard_gap": (
                common_metrics["terminal_summed_squared_error"] - soft_loss
            ),
            "effective_table_displacement": q_displacement(
                initial_q, state.params, condition.representation
            ),
            "gate_diagnostics": _gate_diagnostics(
                state.params, wires, condition.representation
            ),
        }
        for hardening, metrics in (
            ("native", native_metrics),
            ("common", common_metrics),
        ):
            if (
                metrics["exact_all_32"]
                and manifest["first_exact_saved_checkpoint"][hardening] is None
            ):
                manifest["first_exact_saved_checkpoint"][hardening] = update
        trajectory_values = {}
        for label, q in (
            (
                "soft",
                effective_truth_tables(state.params, condition.representation),
            ),
            (
                "native",
                hardened_truth_tables(state.params, condition.representation, "native"),
            ),
            (
                "common",
                hardened_truth_tables(state.params, condition.representation, "common"),
            ),
        ):
            trajectory_values[label] = np.asarray(compiled_trajectory(q))
        trajectory_path = (
            artifact_directory / f"probe_trajectory_update_{update:03d}.npz"
        )
        np.savez_compressed(trajectory_path, **trajectory_values)
        checkpoint_path = artifact_directory / f"checkpoint_update_{update:03d}.npz"
        save_checkpoint(checkpoint_path, state, data_key, checkpoint_metadata)
        manifest["evaluations"].append(record)
        manifest["checkpoints"].append(
            {
                "update": update,
                "checkpoint": _artifact(checkpoint_path, "R002_training_checkpoint"),
                "probe_trajectory": _artifact(
                    trajectory_path, "selected_probe_trajectory"
                ),
            }
        )
        write_json(manifest_path, manifest)

    try:
        save_evaluation(0)
        durations = []
        training_started = time.perf_counter()
        with metrics_path.open("w") as handle:
            for update in range(1, config.optimizer_updates + 1):
                data_key, inputs = sample_training_batch(data_key, config)
                array = np.asarray(inputs)
                stream_digest.update(str(array.dtype).encode())
                stream_digest.update(json.dumps(array.shape).encode())
                stream_digest.update(array.tobytes(order="C"))
                before = state.params
                started = time.perf_counter()
                state, value, gradients, parameter_updates = compiled_step(
                    state, inputs, target, wires
                )
                jax.block_until_ready(state.params)
                duration = time.perf_counter() - started
                durations.append(duration)
                gradient_leaves = jax.tree_util.tree_leaves(gradients)
                metric = {
                    "update": update,
                    "pre_update_soft_loss": float(value),
                    "gradient_max_abs": float(
                        max(jnp.max(jnp.abs(item)) for item in gradient_leaves)
                    ),
                    "gradient_l2": float(
                        jnp.sqrt(
                            sum(jnp.sum(jnp.square(item)) for item in gradient_leaves)
                        )
                    ),
                    "clipped_element_count": int(
                        sum(
                            jnp.count_nonzero(jnp.abs(item) > config.clip_value)
                            for item in gradient_leaves
                        )
                    ),
                    "parameter_update_l2": float(
                        jnp.sqrt(
                            sum(
                                jnp.sum(jnp.square(item))
                                for item in jax.tree_util.tree_leaves(parameter_updates)
                            )
                        )
                    ),
                    "synchronized_seconds": duration,
                    **_transition_metrics(
                        before, state.params, condition.representation
                    ),
                }
                handle.write(json.dumps(metric, sort_keys=True) + "\n")
                handle.flush()
                manifest["current_update"] = update
                if update % 50 == 0:
                    save_evaluation(update)

        manifest["pairing"]["training_stream_sha256"] = stream_digest.hexdigest()
        manifest["training_loop_wall_seconds"] = time.perf_counter() - training_started
        manifest["synchronized_update_seconds"] = float(sum(durations))
        manifest["median_update_seconds"] = float(np.median(durations))
        manifest["final_training_metric"] = json.loads(
            metrics_path.read_text().splitlines()[-1]
        )
        for hardening in ("native", "common"):
            circuit_path = artifact_directory / f"final_{hardening}_circuit.npz"
            ids = gate_ids(state.params, condition.representation, hardening)
            arrays = {}
            for network in ("perceive", "update"):
                for index, values in enumerate(ids[network]):
                    arrays[f"gate_{network}_{index:02d}"] = np.asarray(values)
                for index, (wire_a, wire_b) in enumerate(wires[network]):
                    arrays[f"wire_{network}_{index:02d}_a"] = np.asarray(wire_a)
                    arrays[f"wire_{network}_{index:02d}_b"] = np.asarray(wire_b)
            np.savez_compressed(circuit_path, **arrays)
            manifest.setdefault("artifacts", []).append(
                _artifact(circuit_path, f"final_{hardening}_hard_circuit")
            )
        memory_stats = jax.devices()[0].memory_stats() or {}
        manifest["device_memory"] = {
            name: int(value)
            for name, value in memory_stats.items()
            if name in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"}
            and isinstance(value, (int, np.integer))
        }
        manifest["status"] = "completed"
        manifest["ended_at"] = _utc_now()
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["ended_at"] = _utc_now()
        manifest["failure_reason"] = f"{type(error).__name__}: {error}"
        write_json(manifest_path, manifest)
        raise
    return manifest
