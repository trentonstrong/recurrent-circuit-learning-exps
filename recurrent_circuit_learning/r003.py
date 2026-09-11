"""R003 same-function categorical-mixture diagnostic.

This module intentionally builds on the validated R002 common truth-table
kernel.  R002 checkpoints and wiring artifacts are treated as immutable inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
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

from .difflogic_ca import Params, Wires, bin_op_all_combinations
from .r000 import environment_metadata, sha256_file
from .r002 import (
    GATE_TRUTH_TABLES,
    batch_rollout_q,
    batch_trajectory_q,
    effective_truth_tables,
    gate_ids,
    tree_sha256,
)

T64 = GATE_TRUTH_TABLES.astype(jnp.float64)
NETWORK_LEAF_COUNTS = {"perceive": 3, "update": 16}
EXPECTED_PARAMETER_SHAPES = {
    "perceive": [(16, 8, 16), (16, 4, 16), (16, 2, 16)],
    "update": [(256, 16)] * 10
    + [(128, 16), (64, 16), (32, 16), (16, 16), (8, 16), (8, 16)],
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_config_contract(document: dict[str, Any]) -> None:
    expected = {
        "experiment_id": "R003",
        "protocol_id": "r003_same_function_v1",
        "status": "specified_for_implementation",
    }
    mismatches = {
        key: (document.get(key), value)
        for key, value in expected.items()
        if document.get(key) != value
    }
    source = document.get("source", {})
    recipe = document.get("scientific_recipe", {})
    intervention = document.get("intervention", {})
    step = document.get("diagnostic_step", {})
    frozen = {
        "conditions": source.get("conditions")
        == ["categorical_reference_decay", "categorical_no_decay"],
        "seeds": source.get("seeds") == list(range(16)),
        "updates": source.get("checkpoint_updates") == [0, 250, 500],
        "checkpoint_cases": source.get("expected_logical_checkpoint_cases") == 96,
        "ticks": recipe.get("runtime_ticks") == 20,
        "gate_slots": recipe.get("shared_gate_slots") == 3040,
        "alphas": intervention.get("alphas") == [0, 0.5, 1],
        "learning_rates": step.get("learning_rates") == [0.001, 0.0001],
        "dtype": document.get("numerical", {}).get("primary_dtype") == "float64",
    }
    mismatches.update({key: (False, True) for key, ok in frozen.items() if not ok})
    if mismatches:
        raise ValueError(f"R003 config differs from frozen v1 protocol: {mismatches}")


def probabilities(logits: jax.Array) -> jax.Array:
    return jax.nn.softmax(logits, axis=-1)


def effective_q_from_logits(logits: jax.Array) -> jax.Array:
    return jnp.matmul(
        probabilities(logits), T64, precision=jax.lax.Precision.HIGHEST
    )


def effective_q(params: Params) -> Params:
    return jax.tree_util.tree_map(effective_q_from_logits, params)


def factorized_probabilities(q: jax.Array) -> jax.Array:
    """Independent Bernoulli distribution on the 16 truth tables."""
    log_weights = jnp.sum(
        T64 * jnp.log(q[..., None, :])
        + (1 - T64) * jnp.log1p(-q[..., None, :]),
        axis=-1,
    )
    return jax.nn.softmax(log_weights, axis=-1)


def eligibility_mask(logits: jax.Array, q_margin: float) -> jax.Array:
    p = probabilities(logits)
    q = effective_q_from_logits(logits)
    return (
        jnp.all(jnp.isfinite(logits), axis=-1)
        & jnp.all(jnp.isfinite(p) & (p > 0), axis=-1)
        & jnp.all((q >= q_margin) & (q <= 1 - q_margin), axis=-1)
    )


def replace_logits(
    logits: jax.Array, alpha: float, q_margin: float
) -> tuple[jax.Array, jax.Array]:
    """Move eligible gates toward the factorized representative at fixed q."""
    if alpha == 0:
        return jax.lax.stop_gradient(logits), eligibility_mask(logits, q_margin)
    p = probabilities(logits)
    q = effective_q_from_logits(logits)
    eligible = eligibility_mask(logits, q_margin)
    p_factorized = factorized_probabilities(q)
    p_alpha = (1 - alpha) * p + alpha * p_factorized
    log_p = jnp.log(p_alpha)
    encoded = log_p + (
        jnp.mean(logits, axis=-1, keepdims=True)
        - jnp.mean(log_p, axis=-1, keepdims=True)
    )
    replaced = jnp.where(eligible[..., None], encoded, logits)
    return jax.lax.stop_gradient(replaced), eligible


def replace_tree(
    params: Params, alpha: float, q_margin: float
) -> tuple[Params, Params]:
    replaced = {
        network: [replace_logits(value, alpha, q_margin)[0] for value in leaves]
        for network, leaves in params.items()
    }
    masks = {
        network: [replace_logits(value, alpha, q_margin)[1] for value in leaves]
        for network, leaves in params.items()
    }
    return replaced, masks


def local_geometry(
    logits: jax.Array, h: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return p, J, M, logit gradient, and induced negative q response."""
    p = probabilities(logits)
    q = jnp.matmul(p, T64, precision=jax.lax.Precision.HIGHEST)
    # J[..., r, g] = dq_r / dz_g.
    jacobian = jnp.swapaxes(p[..., :, None] * (T64 - q[..., None, :]), -1, -2)
    matrix = jnp.matmul(
        jacobian, jnp.swapaxes(jacobian, -1, -2),
        precision=jax.lax.Precision.HIGHEST,
    )
    gradient = jnp.einsum("...rg,...r->...g", jacobian, h)
    response = -jnp.einsum("...rs,...s->...r", matrix, h)
    return p, jacobian, matrix, gradient, response


def geometry_tree(params: Params, h: Params) -> dict[str, Params]:
    values: dict[str, Params] = {
        name: {network: [] for network in params}  # type: ignore[assignment]
        for name in ("p", "j", "m", "g", "v")
    }
    for network in params:
        for logits, h_value in zip(params[network], h[network]):
            outputs = local_geometry(logits, h_value)
            for name, value in zip(values, outputs):
                values[name][network].append(value)
    return values


def tree_dot(left: Params, right: Params) -> jax.Array:
    return sum(
        jnp.vdot(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
        )
    )


def tree_scale(tree: Params, scalar: float) -> Params:
    return jax.tree_util.tree_map(lambda value: scalar * value, tree)


def tree_add(left: Params, right: Params) -> Params:
    return jax.tree_util.tree_map(lambda a, b: a + b, left, right)


def tree_sub(left: Params, right: Params) -> Params:
    return jax.tree_util.tree_map(lambda a, b: a - b, left, right)


def tree_flatten_array(tree: Any) -> np.ndarray:
    leaves = [np.asarray(value).reshape(-1) for value in jax.tree_util.tree_leaves(tree)]
    return np.concatenate(leaves) if leaves else np.empty(0)


def comparison(
    actual: Any, reference: Any, *, atol: float, rtol: float
) -> dict[str, Any]:
    actual_array = np.asarray(actual, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    difference = actual_array - reference_array
    allowed = atol + rtol * np.abs(reference_array)
    reference_norm = float(np.linalg.norm(reference_array.ravel()))
    return {
        "ok": bool(np.all(np.abs(difference) <= allowed)),
        "max_abs_error": float(np.max(np.abs(difference), initial=0.0)),
        "relative_l2_error": (
            float(np.linalg.norm(difference.ravel()) / reference_norm)
            if reference_norm
            else (0.0 if not np.any(difference) else None)
        ),
        "atol": atol,
        "rtol": rtol,
    }


def norm_angle(left: Any, right: Any) -> dict[str, Any]:
    a = tree_flatten_array(left).astype(np.float64)
    b = tree_flatten_array(right).astype(np.float64)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0 or b_norm == 0:
        angle = None
        reason = "zero_norm"
    else:
        cosine = float(np.clip(np.dot(a, b) / (a_norm * b_norm), -1, 1))
        angle = float(math.degrees(math.acos(cosine)))
        reason = None
    return {"left_l2": a_norm, "right_l2": b_norm, "angle_degrees": angle, "reason": reason}


def relative_effect(difference: Any, left: Any, right: Any) -> float | None:
    numerator = float(np.linalg.norm(tree_flatten_array(difference).astype(np.float64)))
    denominator = max(
        float(np.linalg.norm(tree_flatten_array(left).astype(np.float64))),
        float(np.linalg.norm(tree_flatten_array(right).astype(np.float64))),
    )
    return numerator / denominator if denominator else None


def _resolve_recorded_artifact(
    record: dict[str, Any], artifact_root: Path, run_id: str
) -> Path:
    candidate = artifact_root / run_id / Path(record["path"]).name
    if not candidate.exists():
        recorded = Path(record["path"])
        if recorded.exists():
            candidate = recorded
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    if candidate.stat().st_size != record["bytes"]:
        raise ValueError(f"artifact byte length mismatch: {candidate}")
    if sha256_file(candidate) != record["sha256"]:
        raise ValueError(f"artifact SHA-256 mismatch: {candidate}")
    return candidate


def _checkpoint_params(path: Path) -> tuple[Params, dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as stored:
        if int(stored["format_version"]) != 2:
            raise ValueError("unsupported R002 checkpoint format")
        if int(stored["param_leaf_count"]) != 19:
            raise ValueError("unexpected categorical checkpoint parameter leaves")
        leaves = [np.asarray(stored[f"param_{index:03d}"]) for index in range(19)]
        expected_shapes = EXPECTED_PARAMETER_SHAPES["perceive"] + EXPECTED_PARAMETER_SHAPES["update"]
        if any(
            value.dtype != np.float32 or value.shape != expected
            for value, expected in zip(leaves, expected_shapes)
        ):
            raise ValueError("unexpected checkpoint parameter dtype or shape")
        params = {
            "perceive": [jnp.asarray(value) for value in leaves[:3]],
            "update": [jnp.asarray(value) for value in leaves[3:]],
        }
        metadata = json.loads(str(stored["metadata_json"]))
        state = {
            "format_version": int(stored["format_version"]),
            "update_index": int(stored["update_index"]),
            "model_key": np.asarray(stored["model_key"]).tolist(),
            "data_key": np.asarray(stored["data_key"]).tolist(),
            "optimizer_leaf_count": int(stored["opt_leaf_count"]),
        }
    return params, metadata, state


def _wires_from_artifact(path: Path) -> Wires:
    with np.load(path, allow_pickle=False) as stored:
        wires: Wires = {"perceive": [], "update": []}
        for network, count in NETWORK_LEAF_COUNTS.items():
            for index in range(count):
                wires[network].append(
                    [
                        jnp.asarray(stored[f"wire_{network}_{index:02d}_a"]),
                        jnp.asarray(stored[f"wire_{network}_{index:02d}_b"]),
                    ]
                )
    return wires


def load_source_case(
    repository_root: Path,
    artifact_root: Path,
    condition: str,
    seed: int,
    update: int,
) -> dict[str, Any]:
    run_id = f"R002_formal_attempt01_seed{seed:02d}_{condition}"
    manifest_path = repository_root / "results" / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {
        "status": "completed",
        "seed": seed,
        "condition": condition,
        "representation": "categorical",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"source manifest mismatch for {key}: {manifest.get(key)!r}")
    checkpoint_entry = next(
        item for item in manifest["checkpoints"] if item["update"] == update
    )
    checkpoint_path = _resolve_recorded_artifact(
        checkpoint_entry["checkpoint"], artifact_root, run_id
    )
    wiring_record = next(
        item for item in manifest["artifacts"]
        if item["kind"] == "final_common_hard_circuit"
    )
    wiring_path = _resolve_recorded_artifact(wiring_record, artifact_root, run_id)
    params, metadata, saved_state = _checkpoint_params(checkpoint_path)
    expected_metadata = {
        "experiment_id": "R002",
        "seed": seed,
        "condition": condition,
        "representation": "categorical",
        "kernel": "common_multilinear_truth_table",
        "config_sha256": manifest["config_sha256"],
        "wiring_sha256": manifest["pairing"]["wiring_sha256"],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint metadata mismatch for {key}")
    if saved_state["update_index"] != update:
        raise ValueError("checkpoint update index mismatch")
    wires = _wires_from_artifact(wiring_path)
    if tree_sha256(wires) != manifest["pairing"]["wiring_sha256"]:
        raise ValueError("loaded wiring tree hash mismatch")
    return {
        "run_id": run_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint_entry["checkpoint"],
        "wiring_path": wiring_path,
        "wiring": wiring_record,
        "params": params,
        "wires": wires,
        "metadata": metadata,
        "saved_state": saved_state,
        "source_code": manifest["code"],
    }


def load_probe(repository_root: Path, document: dict[str, Any]) -> dict[str, Any]:
    path = repository_root / document["probes"]["artifact_path"]
    if sha256_file(path) != document["probes"]["sha256"]:
        raise ValueError("fixed probe SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as stored:
        if set(stored.files) != {"inputs", "target"}:
            raise ValueError("fixed probe array keys mismatch")
        inputs = np.asarray(stored["inputs"])
        target = np.asarray(stored["target"])
    if inputs.shape != (32, 16, 16, 8) or target.shape != inputs.shape:
        raise ValueError("fixed probe shape mismatch")
    if inputs.dtype != np.float32 or target.dtype != np.float32:
        raise ValueError("fixed probe dtype mismatch")
    return {"path": path, "inputs": inputs, "target": target}


def jvp_directions() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(20260911))
    raw = rng.integers(0, 2, size=(2, 2, 16, 16, 8), dtype=np.int8)
    directions = 2 * raw.astype(np.float64) - 1
    norms = np.linalg.norm(directions.reshape(2, -1), axis=1)
    return directions / norms[:, None, None, None, None]


def git_metadata(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository_root, text=True
        ).strip()
    )
    patch = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=repository_root
    )
    return {
        "commit": commit,
        "dirty": dirty,
        "dirty_patch_sha256": hashlib.sha256(patch).hexdigest() if patch else None,
    }


def runtime_metadata(lock_path: Path) -> dict[str, Any]:
    return {
        **environment_metadata(),
        "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "kernel": platform.release(),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }


def _scalar_metrics(tree: Any) -> dict[str, float]:
    flat = tree_flatten_array(tree).astype(np.float64)
    return {
        "l2": float(np.linalg.norm(flat)),
        "max_abs": float(np.max(np.abs(flat), initial=0.0)),
    }


def _tree_eigenvalues(matrix_tree: Params) -> np.ndarray:
    return np.concatenate(
        [
            np.linalg.eigvalsh(np.asarray(value, dtype=np.float64)).reshape(-1, 4)
            for value in jax.tree_util.tree_leaves(matrix_tree)
        ],
        axis=0,
    )


def _tree_layer_records(tree: Params, mask: Params | None = None) -> list[dict[str, Any]]:
    records = []
    for network in ("perceive", "update"):
        for index, value in enumerate(tree[network]):
            array = np.asarray(value)
            record = {
                "network": network,
                "layer": index,
                "shape": list(array.shape),
                **_scalar_metrics(array),
            }
            if mask is not None:
                current = np.asarray(mask[network][index], dtype=bool)
                record["eligible_count"] = int(current.sum())
                record["gate_count"] = int(current.size)
            records.append(record)
    return records


def _save_case_arrays(path: Path, arrays: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _flatten_named(tree: Params, prefix: str, output: dict[str, Any]) -> None:
    for network in ("perceive", "update"):
        for index, value in enumerate(tree[network]):
            output[f"{prefix}_{network}_{index:02d}"] = np.asarray(value)


def _ids_from_q(q: Params) -> Params:
    weights = jnp.array([8, 4, 2, 1], dtype=jnp.int32)
    return jax.tree_util.tree_map(
        lambda value: jnp.sum((value >= 0.5).astype(jnp.int32) * weights, axis=-1), q
    )


def _id_changes(left: Params, right: Params) -> int:
    return int(
        sum(
            np.count_nonzero(np.asarray(a) != np.asarray(b))
            for a, b in zip(
                jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
            )
        )
    )


def _synthetic_controls() -> dict[str, Any]:
    t = np.asarray(T64)
    rank = int(np.linalg.matrix_rank(np.concatenate([np.ones((1, 16)), t.T], axis=0)))
    gate_algebra = np.array_equal(
        t, np.array([[(gate >> (3 - column)) & 1 for column in range(4)] for gate in range(16)])
    )
    uniform = np.full(16, 1 / 16)
    and_or = np.zeros(16); and_or[[1, 7]] = 0.5
    copies = np.zeros(16); copies[[3, 5]] = 0.5
    p1 = 0.8 * and_or + 0.2 * uniform
    p2 = 0.8 * copies + 0.2 * uniform
    z1, z2 = jnp.log(p1), jnp.log(p2)
    q1, q2 = effective_q_from_logits(z1), effective_q_from_logits(z2)
    h = jnp.array([0.2, -0.3, 0.4, -0.1], dtype=jnp.float64)
    _, j1, m1, _, v1 = local_geometry(z1, h)
    _, j2, m2, _, v2 = local_geometry(z2, h)
    auto_j1 = jax.jacrev(effective_q_from_logits)(z1)
    auto_j2 = jax.jacrev(effective_q_from_logits)(z2)
    factorized = factorized_probabilities(q1)
    factorized_again = factorized_probabilities(factorized @ T64)
    gauge_q = effective_q_from_logits(z1 + 1)
    off_fiber = effective_q_from_logits(z1.at[1].add(0.1))
    zero_h = jnp.zeros(4, dtype=jnp.float64)
    zero_v = local_geometry(z1, zero_h)[-1]
    a = jnp.array([0.13, 0.41, 0.79], dtype=jnp.float64)
    b = jnp.array([0.88, 0.37, 0.21], dtype=jnp.float64)
    basis = jnp.stack(((1-a)*(1-b), (1-a)*b, a*(1-b), a*b), axis=-1)
    mapped_extensions = basis @ T64.T
    source_extensions = jax.vmap(lambda left, right: bin_op_all_combinations(left, right))(a, b)
    detached_jacobian = jax.jacrev(lambda value: replace_logits(value, 1.0, 1e-8)[0])(z1)
    checks = {
        "truth_table_rank_5": rank == 5,
        "gate_algebra": gate_algebra,
        "multilinear_extensions": bool(np.allclose(mapped_extensions, source_extensions, atol=1e-15)),
        "example_q": bool(np.allclose(q1, [0.1, 0.5, 0.5, 0.9], atol=1e-15)),
        "example_same_q": bool(np.allclose(q1, q2, atol=1e-15)),
        "example_different_j": not bool(np.allclose(j1, j2)),
        "example_different_m": not bool(np.allclose(m1, m2)),
        "example_different_response": not bool(np.allclose(v1, v2)),
        "jacobian_1_autodiff": bool(np.allclose(j1, auto_j1, atol=1e-12, rtol=1e-10)),
        "jacobian_2_autodiff": bool(np.allclose(j2, auto_j2, atol=1e-12, rtol=1e-10)),
        "gauge_q": bool(np.allclose(q1, gauge_q, atol=1e-15)),
        "factorized_identity": bool(np.allclose(factorized, factorized_again, atol=1e-15)),
        "off_fiber_detected": not bool(np.allclose(q1, off_fiber, atol=1e-12, rtol=1e-10)),
        "zero_h_zero_response": bool(np.array_equal(np.asarray(zero_v), np.zeros(4))),
        "replacement_detached": bool(np.array_equal(np.asarray(detached_jacobian), np.zeros((16, 16)))),
        "post_reset_coordinate_count": int(z1.size) == 16,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "rank": rank, "checks": checks}


@jax.jit
def _loss_and_q_gradient(q: Params, wires: Wires, inputs, target):
    def objective(current):
        predicted = batch_rollout_q(inputs, current, wires, False, 20)
        return jnp.square(predicted[..., 0] - target[..., 0]).sum()

    return jax.value_and_grad(objective)(q)


@jax.jit
def _trajectory(q: Params, wires: Wires, inputs):
    return batch_trajectory_q(inputs, q, wires, False, 20)


@jax.jit
def _visible_outputs(q: Params, wires: Wires, inputs):
    return batch_rollout_q(inputs, q, wires, False, 20)[..., 0]


@jax.jit
def _initial_state_jvp(q: Params, wires: Wires, inputs, direction):
    return jax.jvp(
        lambda current: batch_rollout_q(current, q, wires, False, 20),
        (inputs,),
        (direction,),
    )[1]


@jax.jit
def _output_q_jvp(q: Params, wires: Wires, inputs, direction: Params):
    return jax.jvp(
        lambda current: _visible_outputs(current, wires, inputs),
        (q,),
        (direction,),
    )[1]


@jax.jit
def _full_logit_gradient(params: Params, wires: Wires, inputs, target):
    def objective(current):
        predicted = batch_rollout_q(inputs, effective_q(current), wires, False, 20)
        return jnp.square(predicted[..., 0] - target[..., 0]).sum()

    return jax.grad(objective)(params)


def _trajectory_in_chunks(q: Params, wires: Wires, inputs: jax.Array) -> jax.Array:
    chunks = [_trajectory(q, wires, inputs[start : start + 2]) for start in (0, 2)]
    return jnp.concatenate(chunks, axis=1)


def _first_offending_tick(actual, reference, atol: float, rtol: float) -> int | None:
    difference = np.abs(np.asarray(actual) - np.asarray(reference))
    allowed = atol + rtol * np.abs(np.asarray(reference))
    bad = np.any(difference > allowed, axis=tuple(range(1, difference.ndim)))
    indices = np.flatnonzero(bad)
    return int(indices[0]) if indices.size else None


def _response_from_matrix(matrix: Params, h: Params) -> Params:
    return jax.tree_util.tree_map(
        lambda m, value: -jnp.einsum("...rs,...s->...r", m, value), matrix, h
    )


def _coverage(response: Params, masks: Params) -> dict[str, Any]:
    total = 0.0
    eligible = 0.0
    count = 0
    eligible_count = 0
    for value, mask in zip(
        jax.tree_util.tree_leaves(response), jax.tree_util.tree_leaves(masks)
    ):
        squared = np.square(np.asarray(value, dtype=np.float64)).sum(axis=-1)
        current = np.asarray(mask, dtype=bool)
        total += float(squared.sum())
        eligible += float(squared[current].sum())
        count += current.size
        eligible_count += int(current.sum())
    return {
        "eligible_gates": eligible_count,
        "gate_count": count,
        "eligible_fraction": eligible_count / count,
        "baseline_response_squared_norm": total,
        "eligible_response_squared_norm": eligible,
        "eligible_response_fraction": eligible / total if total else None,
    }


def _normalization_metrics(params: Params, original: Params, masks: Params) -> dict[str, Any]:
    max_sum = 0.0
    min_p = math.inf
    max_p_change = 0.0
    max_mean_change = 0.0
    unchanged_ok = True
    for value, source, mask in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(original),
        jax.tree_util.tree_leaves(masks),
    ):
        p = np.asarray(probabilities(value))
        p0 = np.asarray(probabilities(source))
        current = np.asarray(mask, dtype=bool)
        max_sum = max(max_sum, float(np.max(np.abs(p.sum(axis=-1) - 1))))
        if np.any(current):
            min_p = min(min_p, float(np.min(p[current])))
        max_p_change = max(max_p_change, float(np.max(np.abs(p - p0))))
        max_mean_change = max(
            max_mean_change,
            float(np.max(np.abs(np.asarray(value).mean(-1) - np.asarray(source).mean(-1)))),
        )
        unchanged_ok &= bool(np.array_equal(np.asarray(value)[~current], np.asarray(source)[~current]))
    return {
        "max_probability_sum_error": max_sum,
        "minimum_eligible_probability": min_p if min_p < math.inf else None,
        "max_probability_change": max_p_change,
        "max_logit_mean_change": max_mean_change,
        "ineligible_logits_unchanged": unchanged_ok,
    }


def _common_mismatches(q: Params, reference: Params, ambiguity: float) -> dict[str, Any]:
    mismatches = []
    for network in ("perceive", "update"):
        for layer, (value, source) in enumerate(zip(q[network], reference[network])):
            a = np.asarray(value)
            b = np.asarray(source)
            changed = (a >= 0.5) != (b >= 0.5)
            for location in np.argwhere(changed):
                gate_location = tuple(int(item) for item in location[:-1])
                column = int(location[-1])
                mismatches.append(
                    {
                        "network": network,
                        "layer": layer,
                        "gate_index": list(gate_location),
                        "column": column,
                        "actual_q": float(a[gate_location + (column,)]),
                        "reference_q": float(b[gate_location + (column,)]),
                        "distance_to_half": float(min(abs(a[gate_location + (column,)] - 0.5), abs(b[gate_location + (column,)] - 0.5))),
                    }
                )
    return {
        "count": len(mismatches),
        "rounding_ambiguous_count": sum(item["distance_to_half"] <= ambiguity for item in mismatches),
        "details": mismatches,
    }


def analyze_case(
    source: dict[str, Any],
    probe_inputs: np.ndarray,
    probe_target: np.ndarray,
    directions: np.ndarray,
    document: dict[str, Any],
    artifact_path: Path,
    *,
    full_preflight_checks: bool,
    retain_states: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    tolerances = document["numerical"]
    q_margin = document["intervention"]["eligibility"]["q_margin"]
    inputs64 = jnp.asarray(probe_inputs, dtype=jnp.float64)
    target64 = jnp.asarray(probe_target, dtype=jnp.float64)
    params32 = source["params"]
    params0 = jax.tree_util.tree_map(lambda value: value.astype(jnp.float64), params32)
    wires = source["wires"]
    q0 = effective_q(params0)
    loss0, h0 = _loss_and_q_gradient(q0, wires, inputs64[:2], target64[:2])
    trajectory0 = _trajectory_in_chunks(q0, wires, inputs64[:4])
    outputs0 = _visible_outputs(q0, wires, inputs64[:4])
    jvps0 = [
        _initial_state_jvp(q0, wires, inputs64[:2], jnp.asarray(direction))
        for direction in directions
    ]
    original_ids = gate_ids(params0, "categorical", "native")
    common_ids = _ids_from_q(q0)
    arms: list[dict[str, Any]] = []
    saved_arrays: dict[str, Any] = {}
    _flatten_named(h0, "h_original", saved_arrays)
    baseline_masks = {
        network: [eligibility_mask(value, q_margin) for value in params0[network]]
        for network in params0
    }
    baseline_geometry = geometry_tree(params0, h0)
    coverage = _coverage(baseline_geometry["v"], baseline_masks)
    baseline_output_response = _output_q_jvp(
        q0, wires, inputs64[:4], baseline_geometry["v"]
    )

    fp32_q = effective_truth_tables(params32, "categorical")
    fp32_outputs = batch_rollout_q(
        jnp.asarray(probe_inputs[:4], dtype=jnp.float32), fp32_q, wires, False, 20
    )[..., 0]
    fp32_comparison = comparison(
        np.asarray(outputs0), np.asarray(fp32_outputs), atol=1e-6, rtol=1e-5
    )

    case_controls_ok = True
    for alpha in document["intervention"]["alphas"]:
        params, masks = replace_tree(params0, alpha, q_margin)
        q = effective_q(params)
        trajectory = _trajectory_in_chunks(q, wires, inputs64[:4])
        loss_value, h = _loss_and_q_gradient(q, wires, inputs64[:2], target64[:2])
        jvps = [
            _initial_state_jvp(q, wires, inputs64[:2], jnp.asarray(direction))
            for direction in directions
        ]
        outputs = _visible_outputs(q, wires, inputs64[:4])
        geometry = geometry_tree(params, h)
        v_common = _response_from_matrix(geometry["m"], h0)
        if alpha == 0:
            w_actual = baseline_output_response
            w_common = baseline_output_response
        else:
            w_actual = _output_q_jvp(q, wires, inputs64[:4], geometry["v"])
            w_common = _output_q_jvp(q0, wires, inputs64[:4], v_common)
        q_check = comparison(
            tree_flatten_array(q),
            tree_flatten_array(q0),
            **tolerances["effective_table"],
        )
        state_check = comparison(
            trajectory, trajectory0, **tolerances["recurrent_states"]
        )
        state_check["first_offending_tick"] = _first_offending_tick(
            trajectory, trajectory0, **tolerances["recurrent_states"]
        )
        loss_check = comparison(
            loss_value, loss0, **tolerances["loss"]
        )
        h_check = comparison(
            tree_flatten_array(h),
            tree_flatten_array(h0),
            **tolerances["state_jvp_and_q_gradient"],
        )
        jvp_checks = [
            comparison(value, reference, **tolerances["state_jvp_and_q_gradient"])
            for value, reference in zip(jvps, jvps0)
        ]
        normalization = _normalization_metrics(params, params0, masks)
        normalization_ok = (
            normalization["max_probability_sum_error"]
            <= tolerances["probability_sum"]["atol"]
            and normalization["ineligible_logits_unchanged"]
            and (normalization["minimum_eligible_probability"] or 0) > 0
        )
        common_mismatches = _common_mismatches(
            q, q0, tolerances["common_rounding_ambiguity_distance"]
        )
        common_ok = common_mismatches["count"] == common_mismatches["rounding_ambiguous_count"]
        invariance_ok = all(
            [q_check["ok"], state_check["ok"], loss_check["ok"], h_check["ok"], normalization_ok, common_ok]
            + [item["ok"] for item in jvp_checks]
        )
        case_controls_ok &= invariance_ok

        chain_check = None
        if full_preflight_checks:
            autodiff_g = _full_logit_gradient(params, wires, inputs64[:2], target64[:2])
            chain_check = comparison(
                tree_flatten_array(geometry["g"]),
                tree_flatten_array(autodiff_g),
                **tolerances["full_chain_rule"],
            )
            case_controls_ok &= chain_check["ok"]

        matrix_difference = tree_sub(geometry["m"], baseline_geometry["m"])
        response_difference = tree_sub(v_common, baseline_geometry["v"])
        eigenvalues = _tree_eigenvalues(geometry["m"])
        derivative = float(tree_dot(h, geometry["v"]))
        truth_response = jax.tree_util.tree_map(
            lambda q_value, h_value: -jnp.square(q_value * (1 - q_value)) * h_value,
            q0,
            h0,
        )
        steps = []
        for eta in document["diagnostic_step"]["learning_rates"]:
            stepped = tree_add(params, tree_scale(geometry["g"], -eta))
            q_plus = effective_q(stepped)
            outputs_plus = _visible_outputs(q_plus, wires, inputs64[:4])
            loss_plus, _ = _loss_and_q_gradient(q_plus, wires, inputs64[:2], target64[:2])
            q_delta = tree_sub(q_plus, q)
            output_delta = outputs_plus - outputs
            predicted_q_delta = tree_scale(geometry["v"], eta)
            predicted_output_delta = eta * w_actual
            q_residual = tree_sub(q_delta, predicted_q_delta)
            output_residual = output_delta - predicted_output_delta
            predicted_loss_delta = eta * derivative
            loss_delta = float(loss_plus - loss_value)
            step = {
                "eta": eta,
                "q_delta": _scalar_metrics(q_delta),
                "q_prediction": _scalar_metrics(predicted_q_delta),
                "q_linearization_residual": _scalar_metrics(q_residual),
                "q_linearization_relative_residual": relative_effect(q_residual, q_delta, predicted_q_delta),
                "output_delta": _scalar_metrics(output_delta),
                "output_prediction": _scalar_metrics(predicted_output_delta),
                "output_linearization_residual": _scalar_metrics(output_residual),
                "output_linearization_relative_residual": relative_effect(output_residual, output_delta, predicted_output_delta),
                "loss_delta": loss_delta,
                "predicted_loss_delta": predicted_loss_delta,
                "loss_linearization_residual": loss_delta - predicted_loss_delta,
                "native_gate_id_changes": _id_changes(
                    gate_ids(stepped, "categorical", "native"),
                    gate_ids(params, "categorical", "native"),
                ),
                "common_gate_id_changes": _id_changes(_ids_from_q(q_plus), _ids_from_q(q)),
            }
            steps.append(step)
            prefix = f"alpha_{alpha:g}_eta_{eta:g}"
            _flatten_named(q_delta, f"{prefix}_q_delta", saved_arrays)
            saved_arrays[f"{prefix}_output_delta"] = np.asarray(output_delta)

        prefix = f"alpha_{alpha:g}"
        _flatten_named(q, f"{prefix}_q", saved_arrays)
        _flatten_named(geometry["p"], f"{prefix}_p", saved_arrays)
        _flatten_named(geometry["v"], f"{prefix}_v_actual", saved_arrays)
        _flatten_named(v_common, f"{prefix}_v_common", saved_arrays)
        saved_arrays[f"{prefix}_output_response_actual"] = np.asarray(w_actual)
        saved_arrays[f"{prefix}_output_response_common"] = np.asarray(w_common)
        if retain_states:
            saved_arrays[f"{prefix}_trajectory"] = np.asarray(trajectory)
            for index, value in enumerate(jvps):
                saved_arrays[f"{prefix}_state_jvp_{index}"] = np.asarray(value)
        arm = {
            "alpha": alpha,
            "status": "valid_same_function" if invariance_ok else "failed_control",
            "invariance": {
                "normalization": normalization,
                "normalization_ok": normalization_ok,
                "effective_table": q_check,
                "states": state_check,
                "loss": loss_check,
                "q_gradient": h_check,
                "state_jvps": jvp_checks,
                "common_rounding": common_mismatches,
                "common_rounding_ok": common_ok,
            },
            "full_logit_chain_rule": chain_check,
            "loss": float(loss_value),
            "native_gate_id_changes_from_original": _id_changes(
                gate_ids(params, "categorical", "native"), original_ids
            ),
            "common_gate_id_changes_from_original": _id_changes(_ids_from_q(q), common_ids),
            "matrix": {
                **_scalar_metrics(geometry["m"]),
                "difference_from_original": _scalar_metrics(matrix_difference),
                "relative_difference_from_original": relative_effect(
                    matrix_difference, geometry["m"], baseline_geometry["m"]
                ),
                "eigenvalue_min": float(eigenvalues.min()),
                "eigenvalue_max": float(eigenvalues.max()),
                "numerical_rank_tolerance_1e_12_counts": np.bincount(
                    np.sum(eigenvalues > 1e-12, axis=1), minlength=5
                ).tolist(),
                "layers": _tree_layer_records(geometry["m"]),
            },
            "response": {
                "actual": _scalar_metrics(geometry["v"]),
                "common": _scalar_metrics(v_common),
                "common_difference_from_original": _scalar_metrics(response_difference),
                "common_relative_difference_from_original": relative_effect(
                    response_difference, v_common, baseline_geometry["v"]
                ),
                "actual_to_common": norm_angle(geometry["v"], v_common),
                "d_loss_d_eta": derivative,
                "truth_coordinate_comparator": _scalar_metrics(truth_response),
            },
            "output_response": {
                "actual_loss_inputs": _scalar_metrics(w_actual[:2]),
                "actual_additional_inputs": _scalar_metrics(w_actual[2:]),
                "common_loss_inputs": _scalar_metrics(w_common[:2]),
                "common_additional_inputs": _scalar_metrics(w_common[2:]),
                "actual_difference_from_original": {
                    "loss_inputs": _scalar_metrics(w_actual[:2] - baseline_output_response[:2]),
                    "additional_inputs": _scalar_metrics(w_actual[2:] - baseline_output_response[2:]),
                    "relative": relative_effect(
                        w_actual - baseline_output_response,
                        w_actual,
                        baseline_output_response,
                    ),
                },
                "common_difference_from_original": {
                    "loss_inputs": _scalar_metrics(w_common[:2] - baseline_output_response[:2]),
                    "additional_inputs": _scalar_metrics(w_common[2:] - baseline_output_response[2:]),
                    "relative": relative_effect(
                        w_common - baseline_output_response,
                        w_common,
                        baseline_output_response,
                    ),
                },
            },
            "steps": steps,
        }
        arms.append(arm)

    _flatten_named(baseline_masks, "eligible", saved_arrays)
    saved_arrays["jvp_directions"] = directions
    artifact = _save_case_arrays(artifact_path, saved_arrays)
    return {
        "seed": source["metadata"]["seed"],
        "condition": source["metadata"]["condition"],
        "update": source["saved_state"]["update_index"],
        "status": "passed" if case_controls_ok else "failed",
        "source": {
            key: source[key]
            for key in (
                "run_id", "manifest_path", "manifest_sha256", "checkpoint", "wiring",
                "metadata", "saved_state", "source_code"
            )
        },
        "fp32_to_promoted_fp64_outputs": fp32_comparison,
        "coverage": coverage,
        "eligibility_by_layer": _tree_layer_records(baseline_geometry["v"], baseline_masks),
        "arms": arms,
        "derived_artifact": artifact,
        "source_artifacts_unchanged": {
            "checkpoint": sha256_file(source["checkpoint_path"]) == source["checkpoint"]["sha256"],
            "wiring": sha256_file(source["wiring_path"]) == source["wiring"]["sha256"],
        },
        "wall_seconds": time.perf_counter() - started,
    }


def _summary_markdown(run_id: str, kind: str, cases: list[dict[str, Any]], status: str) -> str:
    valid = sum(arm["status"] == "valid_same_function" for case in cases for arm in case["arms"])
    total_arms = sum(len(case["arms"]) for case in cases)
    native_changes = sum(
        arm["native_gate_id_changes_from_original"]
        for case in cases for arm in case["arms"] if arm["alpha"] == 1
    )
    matrix_effects = [
        arm["matrix"]["difference_from_original"]["l2"]
        for case in cases for arm in case["arms"] if arm["alpha"] == 1 and arm["status"] == "valid_same_function"
    ]
    output_effects = [
        arm["output_response"]["actual_loss_inputs"]["l2"]
        for case in cases for arm in case["arms"] if arm["alpha"] == 1 and arm["status"] == "valid_same_function"
    ]
    return f"""# R003 {kind}: {run_id}

Status: **{status}**.

This is a post-training FP64 mechanism diagnostic on immutable R002 categorical
checkpoints. It is not resumed training and does not establish improved
optimization, causal circuit-size reduction, or naturally traversed neutral paths.

## Coverage and controls

- Checkpoint cases: {len(cases)}
- Valid same-function arm comparisons: {valid}/{total_arms}
- Original-to-factorized native argmax gate changes: {native_changes}
- Factorized-arm matrix-difference L2 range: {min(matrix_effects) if matrix_effects else 'n/a'} to {max(matrix_effects) if matrix_effects else 'n/a'}
- Factorized-arm loss-input output-response L2 range: {min(output_effects) if output_effects else 'n/a'} to {max(output_effects) if output_effects else 'n/a'}

Detailed invariance residuals, geometry, responses, and both independent finite
steps are recorded in `validation.json` and `cases.jsonl`. Derived arrays are
stored outside git under the corresponding artifact directory with hashes.
"""


def run_diagnostic(
    config_path: Path,
    run_id: str,
    repository_root: Path,
    artifact_root: Path,
    results_root: Path,
    lock_path: Path,
    *,
    preflight: bool,
) -> dict[str, Any]:
    if not jax.config.x64_enabled:
        raise RuntimeError("R003 requires JAX_ENABLE_X64=1 before importing JAX")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"R003 requires the GPU runner, got {jax.default_backend()}")
    if jax.config.jax_threefry_partitionable:
        raise RuntimeError("repository determinism requires jax_threefry_partitionable=false")
    config_bytes = config_path.read_bytes()
    document = json.loads(config_bytes)
    validate_config_contract(document)
    label = f"R003_{run_id}"
    result_directory = results_root / label
    output_artifact_directory = artifact_root / label
    if result_directory.exists() or output_artifact_directory.exists():
        raise FileExistsError(f"R003 result or artifact directory already exists: {label}")
    if not preflight:
        passed_preflights = [
            path
            for path in results_root.glob("R003_preflight_attempt*/validation.json")
            if json.loads(path.read_text()).get("status") == "passed"
        ]
        if not passed_preflights:
            raise RuntimeError("the fixed R003 preflight must pass before the formal cohort")
    result_directory.mkdir(parents=True)
    output_artifact_directory.mkdir(parents=True)
    probe = load_probe(repository_root, document)
    directions = jvp_directions()
    direction_path = output_artifact_directory / "state_jvp_directions.npy"
    np.save(direction_path, directions, allow_pickle=False)
    conditions = document["preflight"]["conditions"] if preflight else document["source"]["conditions"]
    seeds = document["preflight"]["seeds"] if preflight else document["source"]["seeds"]
    updates = document["preflight"]["checkpoint_updates"] if preflight else document["source"]["checkpoint_updates"]
    expected_count = document["preflight"]["expected_logical_checkpoint_cases"] if preflight else document["source"]["expected_logical_checkpoint_cases"]
    sources = [
        load_source_case(repository_root, artifact_root, condition, seed, update)
        for seed in seeds for condition in conditions for update in updates
    ]
    if len(sources) != expected_count:
        raise ValueError("source ledger case count mismatch")
    run_started = time.perf_counter()
    manifest = {
        "schema_version": 1,
        "experiment_id": "R003",
        "protocol_id": document["protocol_id"],
        "run_id": label,
        "kind": "preflight" if preflight else "formal_diagnostic",
        "status": "running",
        "started_at": utc_now(),
        "ended_at": None,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "code": git_metadata(repository_root),
        "environment": runtime_metadata(lock_path),
        "probe": {"path": str(probe["path"]), "bytes": probe["path"].stat().st_size, "sha256": sha256_file(probe["path"])},
        "state_jvp_directions": {"path": str(direction_path), "bytes": direction_path.stat().st_size, "sha256": sha256_file(direction_path)},
        "source_ledger": {
            "checkpoint_references": len(sources),
            "wiring_references": len({item["run_id"] for item in sources}),
            "cases": [
                {
                    "run_id": item["run_id"],
                    "update": item["saved_state"]["update_index"],
                    "checkpoint": item["checkpoint"],
                    "wiring": item["wiring"],
                }
                for item in sources
            ],
        },
        "logical_counts": {
            "checkpoint_cases": len(sources),
            "arm_cases": len(sources) * len(document["intervention"]["alphas"]),
            "step_cases": len(sources) * len(document["intervention"]["alphas"]) * len(document["diagnostic_step"]["learning_rates"]),
        },
        "deduplication": {"used": False, "note": "All logical cases were evaluated independently."},
        "failure_reason": None,
    }
    write_json(result_directory / "manifest.json", manifest)
    synthetic = _synthetic_controls()
    cases: list[dict[str, Any]] = []
    cases_path = result_directory / "cases.jsonl"
    try:
        with cases_path.open("w") as handle:
            for source in sources:
                seed = source["metadata"]["seed"]
                condition = source["metadata"]["condition"]
                update = source["saved_state"]["update_index"]
                case = analyze_case(
                    source,
                    probe["inputs"],
                    probe["target"],
                    directions,
                    document,
                    output_artifact_directory / f"case_seed{seed:02d}_{condition}_update{update:03d}.npz",
                    full_preflight_checks=preflight,
                    retain_states=preflight,
                )
                cases.append(case)
                handle.write(json.dumps(case, sort_keys=True) + "\n")
                handle.flush()
        status = "passed" if synthetic["status"] == "passed" and all(case["status"] == "passed" for case in cases) else "failed"
        validation = {
            "schema_version": 1,
            "status": status,
            "synthetic_controls": synthetic,
            "checkpoint_case_count": len(cases),
            "passed_checkpoint_cases": sum(case["status"] == "passed" for case in cases),
            "failed_checkpoint_cases": [
                {"seed": case["seed"], "condition": case["condition"], "update": case["update"]}
                for case in cases if case["status"] != "passed"
            ],
        }
        write_json(result_directory / "validation.json", validation)
        (result_directory / "summary.md").write_text(
            _summary_markdown(label, manifest["kind"], cases, status)
        )
        manifest["status"] = status
        manifest["ended_at"] = utc_now()
        manifest["wall_seconds"] = time.perf_counter() - run_started
        manifest["case_artifacts"] = [case["derived_artifact"] for case in cases]
        memory = jax.devices()[0].memory_stats() or {}
        manifest["device_memory"] = {
            key: int(value) for key, value in memory.items()
            if key in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"} and isinstance(value, (int, np.integer))
        }
        write_json(result_directory / "manifest.json", manifest)
        return {"status": status, "run_id": label, "result_directory": str(result_directory)}
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["ended_at"] = utc_now()
        manifest["failure_reason"] = f"{type(error).__name__}: {error}"
        write_json(result_directory / "manifest.json", manifest)
        raise
