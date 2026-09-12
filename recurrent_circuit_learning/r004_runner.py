"""Exact continuation runner for the frozen R002 training cohort.

R004 changes only the optimizer-update budget.  Parent checkpoints, optimizer
state, random keys, wiring, and all mathematical operations are restored from
the completed R002 runs and then passed to the unchanged R002 train step.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import numpy as np

from .difflogic_ca import TrainState, Wires, make_target, sample_training_batch
from .r000 import environment_metadata, sha256_file
from .r002 import (
    CONDITIONS,
    Condition,
    batch_trajectory_q,
    config_for,
    effective_truth_tables,
    evaluate_params,
    gate_ids,
    hardened_truth_tables,
    init_condition_state,
    load_checkpoint,
    make_optimizer,
    make_train_step,
    save_checkpoint,
    tree_sha256,
)
from .r002_runner import (
    _artifact,
    _comparison,
    _gate_diagnostics,
    _git_metadata,
    _hard_metrics,
    _transition_metrics,
)

PARENT_SWEEP = "formal_attempt01"
PARENT_UPDATE = 500
FINAL_UPDATE = 2000
STRUCTURAL_UPDATES = frozenset((750, 1000, 1250, 1500, 1750, 2000))
EXPECTED_LOCK_SHA256 = (
    "7a8e627cd9851d7426502bc3734bcedcfe48277575aec363877d4e7b717b2125"
)
ORIGINAL_TRAINING_COMMIT = "d220b65da822707cd7db3dc90df69d9250aa0f77"


@dataclass(frozen=True)
class ParentSource:
    manifest_path: Path
    manifest: dict[str, Any]
    checkpoint_path: Path
    checkpoint_record: dict[str, Any]
    exports: dict[str, Path]
    state: TrainState
    data_key: jax.Array
    wires: Wires


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def code_provenance() -> dict[str, Any]:
    """Record dirty state plus the exact R004 implementation source identities."""
    result = _git_metadata()
    repository_root = Path(__file__).resolve().parents[1]
    result["implementation_sources"] = {
        path: {
            "bytes": (repository_root / path).stat().st_size,
            "sha256": sha256_file(repository_root / path),
        }
        for path in (
            "recurrent_circuit_learning/r004_runner.py",
            "recurrent_circuit_learning/r004_analysis.py",
            "scripts/r004.py",
        )
    }
    return result


def validate_config_contract(
    document: dict[str, Any],
) -> tuple[tuple[int, ...], dict[str, Condition]]:
    """Reject changes to fields that define the frozen R004 v1 protocol."""
    expected_top = {
        "schema_version": 1,
        "kind": "training_budget_continuation",
        "experiment_id": "R004",
        "protocol_id": "r004_training_continuation_v1",
        "status": "specified_for_implementation",
    }
    source = document.get("source", {})
    recipe = document.get("scientific_recipe", {})
    resume = document.get("resume", {})
    runtime = document.get("runtime", {})
    evaluation = document.get("evaluation", {})
    structure = document.get("structure", {})
    profile = document.get("runtime_profile", {})
    preflight = document.get("preflight", {})
    analysis = document.get("analysis", {})
    outputs = document.get("outputs", {})
    expected_sections = {
        **expected_top,
        "parent_experiment": (source.get("parent_experiment"), "R002"),
        "parent_sweep": (source.get("parent_sweep"), PARENT_SWEEP),
        "parent_update": (source.get("parent_checkpoint_update"), PARENT_UPDATE),
        "checkpoint_format": (source.get("checkpoint_format_version"), 2),
        "seeds": (recipe.get("seeds"), list(range(16))),
        "run_count": (recipe.get("run_count"), 64),
        "start_update": (recipe.get("start_optimizer_update"), PARENT_UPDATE),
        "final_update": (recipe.get("final_optimizer_update"), FINAL_UPDATE),
        "first_new_update": (recipe.get("first_new_optimizer_update"), 501),
        "additional_updates": (recipe.get("additional_updates_per_run"), 1500),
        "total_updates": (recipe.get("total_additional_updates"), 96000),
        "learning_rate": (recipe.get("learning_rate"), 0.05),
        "adam_b1": (recipe.get("adam_b1"), 0.9),
        "adam_b2": (recipe.get("adam_b2"), 0.99),
        "adam_eps": (recipe.get("adam_eps"), 1e-8),
        "adam_eps_root": (recipe.get("adam_eps_root"), 0),
        "clip_value": (recipe.get("clip_value"), 100),
        "batch_size": (recipe.get("batch_size"), 2),
        "runtime_ticks": (recipe.get("runtime_ticks"), 20),
        "grid_size": (recipe.get("grid_size"), [16, 16]),
        "channels": (recipe.get("channels"), 8),
        "kernel": (recipe.get("kernel"), "common_multilinear_truth_table"),
        "loss": (
            recipe.get("loss"),
            "summed_terminal_channel_0_squared_error",
        ),
        "early_stopping": (recipe.get("early_stopping"), False),
        "new_trials": (recipe.get("new_independent_trials"), False),
        "restore": (
            resume.get("restore"),
            [
                "parameters",
                "complete_optimizer_state",
                "model_key",
                "data_key",
                "global_update_index",
                "actual_wiring",
            ],
        ),
        "new_optimizer": (resume.get("initialize_new_optimizer_moments"), False),
        "atomic": (resume.get("atomic_checkpoint_write"), True),
        "lock_hash": (runtime.get("expected_lock_sha256"), EXPECTED_LOCK_SHA256),
        "backend": (runtime.get("backend"), "gpu"),
        "x64": (runtime.get("jax_enable_x64"), False),
        "threefry": (runtime.get("jax_threefry_partitionable"), False),
        "evaluation_updates": (
            evaluation.get("optimizer_updates"),
            list(range(500, 2001, 50)),
        ),
        "checkpoint_interval": (
            evaluation.get("state_and_optimizer_checkpoint_interval"),
            50,
        ),
        "structure_updates": (
            structure.get("optimizer_updates"),
            [500, 750, 1000, 1250, 1500, 1750, 2000],
        ),
        "profile_updates": (
            profile.get("optimizer_updates"),
            [500, 1000, 1500, 2000],
        ),
        "profile_horizon": (profile.get("horizon"), 256),
        "historical_replay": (
            (
                preflight.get("historical_replay_from_update"),
                preflight.get("historical_replay_through_update"),
            ),
            (450, 500),
        ),
        "bootstrap": (
            analysis.get("bootstrap", {}).get("index_shape"),
            [100000, 16],
        ),
        "run_pattern": (
            outputs.get("run_results_pattern"),
            "results/R004_{sweep_id}_seed{seed:02d}_{condition}",
        ),
    }
    mismatches: dict[str, tuple[Any, Any]] = {}
    for name, expected in expected_top.items():
        actual = document.get(name)
        if actual != expected:
            mismatches[name] = (actual, expected)
    for name, pair in expected_sections.items():
        if name in expected_top:
            continue
        actual, expected = pair
        if actual != expected:
            mismatches[name] = (actual, expected)

    declared = {
        item["name"]: Condition(
            item["name"], item["representation"], item["weight_decay"]
        )
        for item in document.get("conditions", [])
    }
    if declared != CONDITIONS:
        mismatches["conditions"] = (declared, CONDITIONS)
    if mismatches:
        raise ValueError(f"R004 config differs from frozen v1 protocol: {mismatches}")
    return tuple(recipe["seeds"]), declared


def batch_sha256(batch: Any) -> str:
    array = np.asarray(batch)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def continuation_batch_hashes_sha256(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    expected = 501
    for record in records:
        update = int(record["global_update"])
        if update != expected:
            raise ValueError(f"non-contiguous continuation metrics at update {update}")
        digest.update(f"{update}:{record['training_batch_sha256']}\n".encode("ascii"))
        expected += 1
    return digest.hexdigest()


def _record_for_name(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    records = list(manifest.get("artifacts", []))
    for checkpoint in manifest.get("checkpoints", []):
        records.extend((checkpoint["checkpoint"], checkpoint["probe_trajectory"]))
    matches = [record for record in records if Path(record["path"]).name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one parent artifact named {name!r}")
    return matches[0]


def _resolve_parent_artifact(
    record: dict[str, Any], artifact_root: Path, run_id: str
) -> Path:
    relocated = artifact_root / run_id / Path(record["path"]).name
    path = relocated if relocated.is_file() else Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"parent artifact identity mismatch: {path}")
    return path


def _load_wires(export_path: Path) -> Wires:
    with np.load(export_path, allow_pickle=False) as stored:
        return {
            network: [
                [
                    jnp.asarray(stored[f"wire_{network}_{index:02d}_a"]),
                    jnp.asarray(stored[f"wire_{network}_{index:02d}_b"]),
                ]
                for index in range(3 if network == "perceive" else 16)
            ]
            for network in ("perceive", "update")
        }


def _export_gate_ids(export_path: Path) -> dict[str, list[np.ndarray]]:
    with np.load(export_path, allow_pickle=False) as stored:
        return {
            network: [
                stored[f"gate_{network}_{index:02d}"]
                for index in range(3 if network == "perceive" else 16)
            ]
            for network in ("perceive", "update")
        }


def _trees_equal(left: Any, right: Any) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    return (
        left_tree == right_tree
        and len(left_leaves) == len(right_leaves)
        and all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(left_leaves, right_leaves)
        )
    )


def _validate_tree_layout(actual: Any, template: Any, label: str) -> None:
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    template_leaves, template_tree = jax.tree_util.tree_flatten(template)
    if actual_tree != template_tree or len(actual_leaves) != len(template_leaves):
        raise ValueError(f"parent {label} tree structure mismatch")
    for index, (value, expected) in enumerate(zip(actual_leaves, template_leaves)):
        value_array = np.asarray(value)
        expected_array = np.asarray(expected)
        if (
            value_array.shape != expected_array.shape
            or value_array.dtype != expected_array.dtype
        ):
            raise ValueError(f"parent {label} leaf {index} layout mismatch")


def load_parent(
    repository_root: Path,
    artifact_root: Path,
    seed: int,
    condition: Condition,
    *,
    update: int = PARENT_UPDATE,
) -> ParentSource:
    run_id = f"R002_{PARENT_SWEEP}_seed{seed:02d}_{condition.name}"
    manifest_path = repository_root / "results" / run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("experiment_id") != "R002"
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "completed"
        or manifest.get("current_update") != PARENT_UPDATE
        or manifest.get("seed") != seed
        or manifest.get("condition") != condition.name
        or manifest.get("representation") != condition.representation
        or manifest.get("weight_decay") != condition.weight_decay
    ):
        raise ValueError(f"invalid R002 parent manifest: {run_id}")
    expected_environment = {
        "backend": "gpu",
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "numpy": "2.4.6",
        "optax": "0.2.8",
        "python": "3.11.6",
        "jax_enable_x64": False,
        "jax_threefry_partitionable": False,
        "lock_sha256": EXPECTED_LOCK_SHA256,
        "xla_flags": " --xla_gpu_deterministic_ops=true",
    }
    if manifest.get("code", {}).get("commit") != ORIGINAL_TRAINING_COMMIT:
        raise ValueError(f"R002 parent training commit mismatch: {run_id}")
    for name, expected in expected_environment.items():
        if manifest.get("environment", {}).get(name) != expected:
            raise ValueError(f"R002 parent environment mismatch for {name}: {run_id}")
    # R002 hashes the original config bytes. Validate against the checked-in source
    # as the JSON re-encoding above need not preserve whitespace.
    parent_config = (
        repository_root / "configs/experiments/r002_paired_gate_coordinates.json"
    )
    if manifest["config_sha256"] != hashlib.sha256(
        parent_config.read_bytes()
    ).hexdigest() or manifest["config"] != json.loads(parent_config.read_text()):
        raise ValueError(f"R002 parent config identity mismatch: {run_id}")

    checkpoint_name = f"checkpoint_update_{update:03d}.npz"
    checkpoint_record = _record_for_name(manifest, checkpoint_name)
    checkpoint_path = _resolve_parent_artifact(checkpoint_record, artifact_root, run_id)
    export_paths = {
        hardening: _resolve_parent_artifact(
            _record_for_name(manifest, f"final_{hardening}_circuit.npz"),
            artifact_root,
            run_id,
        )
        for hardening in ("native", "common")
    }
    wires = _load_wires(export_paths["common"])
    native_wires = _load_wires(export_paths["native"])
    if not _trees_equal(wires, native_wires):
        raise ValueError(f"parent exports disagree on wiring: {run_id}")
    if tree_sha256(wires) != manifest["pairing"]["wiring_sha256"]:
        raise ValueError(f"parent wiring hash mismatch: {run_id}")

    config = config_for(seed, condition, optimizer_updates=FINAL_UPDATE)
    optimizer = make_optimizer(config)
    template, _, _ = init_condition_state(seed, condition, optimizer)
    expected_metadata = {
        "experiment_id": "R002",
        "seed": seed,
        "condition": condition.name,
        "representation": condition.representation,
        "kernel": "common_multilinear_truth_table",
        "config_sha256": manifest["config_sha256"],
        "wiring_sha256": manifest["pairing"]["wiring_sha256"],
    }
    state, data_key, _ = load_checkpoint(checkpoint_path, template, expected_metadata)
    if int(state.update_index) != update:
        raise ValueError(f"parent checkpoint update mismatch: {run_id}")
    _validate_tree_layout(state.params, template.params, "parameter")
    _validate_tree_layout(state.opt_state, template.opt_state, "optimizer")
    if np.asarray(state.key).shape != np.asarray(template.key).shape:
        raise ValueError(f"parent model key layout mismatch: {run_id}")
    if np.asarray(data_key).shape != np.asarray(template.key).shape:
        raise ValueError(f"parent data key layout mismatch: {run_id}")
    if update == PARENT_UPDATE:
        for hardening, path in export_paths.items():
            expected_ids = _export_gate_ids(path)
            actual_ids = gate_ids(state.params, condition.representation, hardening)
            if not _trees_equal(actual_ids, expected_ids):
                raise ValueError(f"parent gate export mismatch: {run_id}/{hardening}")
    return ParentSource(
        manifest_path,
        manifest,
        checkpoint_path,
        checkpoint_record,
        export_paths,
        state,
        data_key,
        wires,
    )


def checkpoint_metadata(
    *,
    config_sha256: str,
    seed: int,
    condition: Condition,
    parent: ParentSource,
) -> dict[str, Any]:
    return {
        "experiment_id": "R004",
        "protocol_id": "r004_training_continuation_v1",
        "seed": seed,
        "condition": condition.name,
        "representation": condition.representation,
        "kernel": "common_multilinear_truth_table",
        "config_sha256": config_sha256,
        "wiring_sha256": parent.manifest["pairing"]["wiring_sha256"],
        "parent_run_id": parent.manifest["run_id"],
        "parent_manifest_sha256": sha256_file(parent.manifest_path),
        "parent_checkpoint_sha256": parent.checkpoint_record["sha256"],
        "parent_checkpoint_bytes": parent.checkpoint_record["bytes"],
        "parent_training_stream_sha256": parent.manifest["pairing"][
            "training_stream_sha256"
        ],
    }


def save_checkpoint_atomic(
    path: Path,
    state: TrainState,
    data_key: jax.Array,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    save_checkpoint(temporary, state, data_key, metadata)
    temporary.replace(path)


def _save_circuit(
    path: Path, state: TrainState, wires: Wires, condition: Condition, hardening: str
) -> None:
    ids = gate_ids(state.params, condition.representation, hardening)
    arrays: dict[str, Any] = {}
    for network in ("perceive", "update"):
        for index, values in enumerate(ids[network]):
            arrays[f"gate_{network}_{index:02d}"] = np.asarray(values)
        for index, (wire_a, wire_b) in enumerate(wires[network]):
            arrays[f"wire_{network}_{index:02d}_a"] = np.asarray(wire_a)
            arrays[f"wire_{network}_{index:02d}_b"] = np.asarray(wire_b)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    updates = [record["global_update"] for record in records]
    if updates != list(range(501, 501 + len(records))):
        raise ValueError("R004 metrics are not a unique contiguous continuation")
    return records


def _write_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    temporary.replace(path)


def _archive_superseded_tail(
    result_directory: Path,
    records: list[dict[str, Any]],
    durable_update: int,
) -> list[dict[str, Any]]:
    retained = [
        record for record in records if record["global_update"] <= durable_update
    ]
    tail = [record for record in records if record["global_update"] > durable_update]
    if tail:
        history_path = result_directory / "attempt_history.jsonl"
        with history_path.open("a") as handle:
            for record in tail:
                handle.write(
                    json.dumps(
                        {
                            "status": "superseded_uncheckpointed_tail",
                            "superseded_at": utc_now(),
                            "metric": record,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return retained


def _batch_hashes_from_parent(
    parent_key: jax.Array, config: Any, through_update: int
) -> tuple[list[str], jax.Array]:
    key = parent_key
    hashes = []
    for _update in range(501, through_update + 1):
        key, inputs = sample_training_batch(key, config)
        hashes.append(batch_sha256(inputs))
    return hashes, key


def _validate_metric_batch_history(
    parent: ParentSource,
    config: Any,
    records: list[dict[str, Any]],
    data_key: jax.Array,
) -> None:
    expected_hashes, expected_key = _batch_hashes_from_parent(
        parent.data_key, config, 500 + len(records)
    )
    actual_hashes = [record["training_batch_sha256"] for record in records]
    if actual_hashes != expected_hashes or not np.array_equal(
        np.asarray(data_key), np.asarray(expected_key)
    ):
        raise ValueError("R004 metric batch history or restored data key mismatch")


def _baseline_record(
    parent: ParentSource,
    condition: Condition,
    config: Any,
    compiled_evaluation: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    soft, native, common = jax.device_get(compiled_evaluation(parent.state.params))
    with np.load(Path(parent.manifest["probe"]["path"]), allow_pickle=False) as stored:
        target = stored["target"]
    parent_record = parent.manifest["evaluations"][-1]
    soft_array = np.asarray(soft)
    soft_loss = float(np.square(soft_array[..., 0] - target[..., 0]).sum())
    native_metrics = _hard_metrics(np.asarray(native), target)
    common_metrics = _hard_metrics(np.asarray(common), target)
    controls = {
        "soft": _comparison(
            soft_loss,
            parent_record["soft_terminal_summed_squared_error"],
            1e-6,
            1e-4,
        ),
        "native_bit_errors_exact": native_metrics["bit_errors"]
        == parent_record["native"]["bit_errors"],
        "common_bit_errors_exact": common_metrics["bit_errors"]
        == parent_record["common"]["bit_errors"],
        "native_perfect_grids_exact": native_metrics["perfect_grid_count"]
        == parent_record["native"]["perfect_grid_count"],
        "common_perfect_grids_exact": common_metrics["perfect_grid_count"]
        == parent_record["common"]["perfect_grid_count"],
    }
    if not all(
        value["ok"] if isinstance(value, dict) else value for value in controls.values()
    ):
        raise ValueError(f"R004 baseline replay mismatch: {parent.manifest['run_id']}")
    return (
        {
            "global_update": 500,
            "continuation_update": 0,
            "kind": "R004_parent_boundary_replay",
            "parent_evaluation_by_reference": {
                "manifest": str(parent.manifest_path),
                "update": 500,
            },
            "soft_terminal_summed_squared_error": soft_loss,
            "native": native_metrics,
            "common": common_metrics,
            "native_soft_hard_gap": native_metrics["terminal_summed_squared_error"]
            - soft_loss,
            "common_soft_hard_gap": common_metrics["terminal_summed_squared_error"]
            - soft_loss,
            "gate_diagnostics": _gate_diagnostics(
                parent.state.params, parent.wires, condition.representation
            ),
        },
        controls,
    )


def _probe_path(repository_root: Path, document: dict[str, Any]) -> Path:
    path = repository_root / document["evaluation"]["original_probe_path"]
    if sha256_file(path) != document["evaluation"]["original_probe_sha256"]:
        raise ValueError("R004 original probe identity mismatch")
    return path


def _validate_runtime(document: dict[str, Any], gpu_lock_path: Path) -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"R004 requires the GPU runner, got {jax.default_backend()}")
    if jax.config.x64_enabled or jax.config.jax_threefry_partitionable:
        raise RuntimeError(
            "R004 requires X64 disabled and legacy Threefry partitioning"
        )
    if sha256_file(gpu_lock_path) != EXPECTED_LOCK_SHA256:
        raise ValueError("R004 GPU lock identity mismatch")
    actual = environment_metadata()
    for name, expected in document["runtime"]["expected_versions"].items():
        actual_value = actual["python"].split()[0] if name == "python" else actual[name]
        if actual_value != expected:
            raise ValueError(
                f"R004 runtime version mismatch for {name}: "
                f"{actual_value!r} != {expected!r}"
            )
    if os.environ.get("XLA_FLAGS") != document["runtime"]["xla_flags"]:
        raise ValueError("R004 XLA_FLAGS differs from the frozen runtime")


def run_condition(
    config_path: Path,
    seed: int,
    condition: Condition,
    result_directory: Path,
    artifact_directory: Path,
    gpu_lock_path: Path,
    *,
    parent_artifact_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Continue one R002 parent from global update 500 through 2,000."""
    repository_root = config_path.resolve().parents[2]
    config_bytes = config_path.read_bytes()
    document = json.loads(config_bytes)
    seeds, declared = validate_config_contract(document)
    _validate_runtime(document, gpu_lock_path)
    if seed not in seeds or declared.get(condition.name) != condition:
        raise ValueError("seed or condition is outside the R004 cohort")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config = config_for(seed, condition, optimizer_updates=FINAL_UPDATE)
    parent = load_parent(repository_root, parent_artifact_root, seed, condition)
    metadata = checkpoint_metadata(
        config_sha256=config_hash,
        seed=seed,
        condition=condition,
        parent=parent,
    )
    probe_path = _probe_path(repository_root, document)
    with np.load(probe_path, allow_pickle=False) as stored:
        probe_inputs = jnp.asarray(stored["inputs"])
        probe_target = np.asarray(stored["target"])

    if result_directory.exists() or artifact_directory.exists():
        if (
            not resume
            or not result_directory.is_dir()
            or not artifact_directory.is_dir()
        ):
            raise FileExistsError("R004 result or artifact directory already exists")
    else:
        result_directory.mkdir(parents=True)
        artifact_directory.mkdir(parents=True)

    optimizer = make_optimizer(config)
    train_step = make_train_step(config, condition.representation, optimizer)
    target = make_target(config)
    _, compile_inputs = sample_training_batch(parent.data_key, config)
    compile_started = time.perf_counter()
    compiled_step = train_step.lower(
        parent.state, compile_inputs, target, parent.wires
    ).compile()
    compile_seconds = time.perf_counter() - compile_started
    evaluation = jax.jit(
        lambda params: (
            evaluate_params(
                probe_inputs, params, parent.wires, condition.representation, config
            ),
            evaluate_params(
                probe_inputs,
                params,
                parent.wires,
                condition.representation,
                config,
                "native",
            ),
            evaluate_params(
                probe_inputs,
                params,
                parent.wires,
                condition.representation,
                config,
                "common",
            ),
        )
    )
    evaluation_started = time.perf_counter()
    compiled_evaluation = evaluation.lower(parent.state.params).compile()
    trajectory_function = jax.jit(
        lambda q: batch_trajectory_q(
            probe_inputs[:1], q, parent.wires, config.periodic, config.runtime_ticks
        )
    )
    q_template = effective_truth_tables(parent.state.params, condition.representation)
    compiled_trajectory = trajectory_function.lower(q_template).compile()
    evaluation_compile_seconds = time.perf_counter() - evaluation_started

    manifest_path = result_directory / "manifest.json"
    metrics_path = result_directory / "metrics.jsonl"
    state = parent.state
    data_key = parent.data_key
    records: list[dict[str, Any]] = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        identity = {
            "experiment_id": "R004",
            "run_id": result_directory.name,
            "seed": seed,
            "condition": condition.name,
            "config_sha256": config_hash,
            "parent_checkpoint_sha256": parent.checkpoint_record["sha256"],
        }
        for name, expected in identity.items():
            actual = (
                manifest.get("lineage", {}).get(name)
                if name == "parent_checkpoint_sha256"
                else manifest.get(name)
            )
            if actual != expected:
                raise ValueError(f"R004 resume identity mismatch for {name}")
        if manifest["status"] == "completed":
            if manifest["current_update"] != FINAL_UPDATE:
                raise ValueError("completed R004 manifest has wrong endpoint")
            return manifest
        checkpoints = manifest.get("checkpoints", [])
        if checkpoints:
            latest = max(checkpoints, key=lambda item: item["update"])
            path = artifact_directory / Path(latest["checkpoint"]["path"]).name
            if (
                not path.is_file()
                or path.stat().st_size != latest["checkpoint"]["bytes"]
                or sha256_file(path) != latest["checkpoint"]["sha256"]
            ):
                raise ValueError("latest R004 checkpoint identity mismatch")
            state, data_key, _ = load_checkpoint(path, parent.state, metadata)
        records = _read_metrics(metrics_path)
        records = _archive_superseded_tail(
            result_directory, records, int(state.update_index)
        )
        _write_metrics(metrics_path, records)
        _validate_metric_batch_history(parent, config, records, data_key)
        manifest["attempts"].append(
            {"started_at": utc_now(), "resumed_from_update": int(state.update_index)}
        )
        manifest["status"] = "running"
        manifest["failure_reason"] = None
    else:
        baseline, baseline_controls = _baseline_record(
            parent, condition, config, compiled_evaluation
        )
        manifest = {
            "schema_version": 1,
            "experiment_id": "R004",
            "protocol_id": "r004_training_continuation_v1",
            "run_id": result_directory.name,
            "status": "running",
            "started_at": utc_now(),
            "ended_at": None,
            "current_update": PARENT_UPDATE,
            "seed": seed,
            "condition": condition.name,
            "representation": condition.representation,
            "weight_decay": condition.weight_decay,
            "config": document,
            "config_sha256": config_hash,
            "code": code_provenance(),
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
            "lineage": {
                "parent_run_id": parent.manifest["run_id"],
                "parent_manifest": _artifact(
                    parent.manifest_path, "R002_parent_manifest"
                ),
                "parent_checkpoint": parent.checkpoint_record,
                "parent_checkpoint_sha256": parent.checkpoint_record["sha256"],
                "parent_training_stream_sha256": parent.manifest["pairing"][
                    "training_stream_sha256"
                ],
                "wiring_sha256": parent.manifest["pairing"]["wiring_sha256"],
                "parent_code": parent.manifest["code"],
            },
            "probe": _artifact(probe_path, "R001_fixed_evaluation_set"),
            "parent_baseline": parent.manifest["evaluations"][-1],
            "evaluations": [baseline],
            "baseline_replay_controls": baseline_controls,
            "checkpoints": [],
            "structural_exports": [
                {
                    "global_update": 500,
                    "kind": "R002_parent_export_by_reference",
                    "hardening": hardening,
                    "artifact": _artifact(path, f"parent_{hardening}_hard_circuit"),
                }
                for hardening, path in parent.exports.items()
            ],
            "attempts": [{"started_at": utc_now(), "resumed_from_update": 500}],
            "training_compile_seconds": compile_seconds,
            "evaluation_compile_seconds": evaluation_compile_seconds,
            "continuation_batch_hashes_sha256": None,
            "failure_reason": None,
        }
        _write_metrics(metrics_path, records)
    write_json(manifest_path, manifest)

    def save_evaluation(update: int) -> None:
        nonlocal state, data_key
        state_identity = tree_sha256(
            (state.params, state.opt_state, state.key, state.update_index, data_key)
        )
        soft, native, common = jax.device_get(compiled_evaluation(state.params))
        soft_array = np.asarray(soft)
        soft_loss = float(np.square(soft_array[..., 0] - probe_target[..., 0]).sum())
        native_metrics = _hard_metrics(np.asarray(native), probe_target)
        common_metrics = _hard_metrics(np.asarray(common), probe_target)
        evaluation_record = {
            "global_update": update,
            "continuation_update": update - 500,
            "kind": "R004_checkpoint_evaluation",
            "soft_terminal_summed_squared_error": soft_loss,
            "native": native_metrics,
            "common": common_metrics,
            "native_soft_hard_gap": native_metrics["terminal_summed_squared_error"]
            - soft_loss,
            "common_soft_hard_gap": common_metrics["terminal_summed_squared_error"]
            - soft_loss,
            "gate_diagnostics": _gate_diagnostics(
                state.params, parent.wires, condition.representation
            ),
        }
        trajectory_values = {
            label: np.asarray(compiled_trajectory(q))
            for label, q in (
                (
                    "soft",
                    effective_truth_tables(state.params, condition.representation),
                ),
                (
                    "native",
                    hardened_truth_tables(
                        state.params, condition.representation, "native"
                    ),
                ),
                (
                    "common",
                    hardened_truth_tables(
                        state.params, condition.representation, "common"
                    ),
                ),
            )
        }
        trajectory_path = (
            artifact_directory / f"probe_trajectory_update_{update:04d}.npz"
        )
        trajectory_temp = trajectory_path.with_name(f".{trajectory_path.stem}.tmp.npz")
        with trajectory_temp.open("wb") as handle:
            np.savez_compressed(handle, **trajectory_values)
        trajectory_temp.replace(trajectory_path)
        checkpoint_path = artifact_directory / f"checkpoint_update_{update:04d}.npz"
        save_checkpoint_atomic(checkpoint_path, state, data_key, metadata)
        checkpoint_record = {
            "update": update,
            "checkpoint": _artifact(checkpoint_path, "R004_training_checkpoint"),
            "probe_trajectory": _artifact(trajectory_path, "selected_probe_trajectory"),
        }
        manifest["evaluations"] = [
            item for item in manifest["evaluations"] if item["global_update"] != update
        ] + [evaluation_record]
        manifest["checkpoints"] = [
            item for item in manifest["checkpoints"] if item["update"] != update
        ] + [checkpoint_record]
        if update in STRUCTURAL_UPDATES:
            for hardening in ("common", "native"):
                circuit_path = artifact_directory / (
                    f"circuit_update_{update:04d}_{hardening}.npz"
                )
                _save_circuit(circuit_path, state, parent.wires, condition, hardening)
                manifest["structural_exports"] = [
                    item
                    for item in manifest["structural_exports"]
                    if not (
                        item["global_update"] == update
                        and item["hardening"] == hardening
                    )
                ] + [
                    {
                        "global_update": update,
                        "hardening": hardening,
                        "artifact": _artifact(
                            circuit_path, f"R004_{hardening}_hard_circuit"
                        ),
                    }
                ]
        if (
            tree_sha256(
                (state.params, state.opt_state, state.key, state.update_index, data_key)
            )
            != state_identity
        ):
            raise RuntimeError("evaluation or export mutated live R004 state")
        manifest["current_update"] = update
        manifest["continuation_batch_hashes_sha256"] = continuation_batch_hashes_sha256(
            records
        )
        write_json(manifest_path, manifest)

    durations: list[float] = []
    loop_started = time.perf_counter()
    try:
        start_update = int(state.update_index) + 1
        with metrics_path.open("a") as handle:
            for update in range(start_update, FINAL_UPDATE + 1):
                data_key, inputs = sample_training_batch(data_key, config)
                batch_hash = batch_sha256(inputs)
                before = state.params
                started = time.perf_counter()
                state, value, gradients, parameter_updates = compiled_step(
                    state, inputs, target, parent.wires
                )
                jax.block_until_ready(state.params)
                duration = time.perf_counter() - started
                if int(state.update_index) != update:
                    raise RuntimeError("R004 global update counter drift")
                gradient_leaves = jax.tree_util.tree_leaves(gradients)
                metric = {
                    "global_update": update,
                    "continuation_update": update - 500,
                    "training_batch_sha256": batch_hash,
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
                if not np.isfinite(metric["pre_update_soft_loss"]):
                    raise FloatingPointError(f"nonfinite R004 loss at update {update}")
                records.append(metric)
                handle.write(json.dumps(metric, sort_keys=True) + "\n")
                handle.flush()
                durations.append(duration)
                manifest["current_update"] = update
                if update % 50 == 0:
                    save_evaluation(update)

        manifest["continuation_batch_hashes_sha256"] = continuation_batch_hashes_sha256(
            records
        )
        manifest["training_loop_wall_seconds"] = manifest.get(
            "training_loop_wall_seconds", 0.0
        ) + (time.perf_counter() - loop_started)
        manifest["synchronized_update_seconds"] = manifest.get(
            "synchronized_update_seconds", 0.0
        ) + sum(durations)
        manifest["final_training_metric"] = records[-1]
        manifest["metrics"] = _artifact(metrics_path, "R004_per_update_metrics")
        manifest["status"] = "completed"
        manifest["ended_at"] = utc_now()
        memory_stats = jax.devices()[0].memory_stats() or {}
        manifest["device_memory"] = {
            name: int(value)
            for name, value in memory_stats.items()
            if name in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"}
            and isinstance(value, (int, np.integer))
        }
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["ended_at"] = utc_now()
        manifest["failure_reason"] = f"{type(error).__name__}: {error}"
        write_json(manifest_path, manifest)
        raise
    return manifest


def compare_states(left: TrainState, right: TrainState) -> dict[str, bool]:
    return {
        "parameters": _trees_equal(left.params, right.params),
        "optimizer": _trees_equal(left.opt_state, right.opt_state),
        "model_key": np.array_equal(np.asarray(left.key), np.asarray(right.key)),
        "update_index": int(left.update_index) == int(right.update_index),
    }


def run_preflight(
    config_path: Path,
    run_id: str,
    result_directory: Path,
    artifact_directory: Path,
    gpu_lock_path: Path,
    parent_artifact_root: Path,
) -> dict[str, Any]:
    """Run the fixed seed-0 restoration, resume, and cohort-boundary checks."""
    if result_directory.exists() or artifact_directory.exists():
        raise FileExistsError("R004 preflight output already exists")
    config_bytes = config_path.read_bytes()
    document = json.loads(config_bytes)
    seeds, conditions = validate_config_contract(document)
    _validate_runtime(document, gpu_lock_path)
    repository_root = config_path.resolve().parents[2]
    result_directory.mkdir(parents=True)
    artifact_directory.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "R004",
        "kind": "R004_preflight",
        "run_id": run_id,
        "status": "running",
        "started_at": utc_now(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "code": code_provenance(),
        "conditions": {},
        "cohort_boundary": {},
        "failure_reason": None,
    }
    report_path = result_directory / "validation.json"
    write_json(report_path, report)
    try:
        for condition in conditions.values():
            config_old = config_for(0, condition, optimizer_updates=500)
            config_new = config_for(0, condition, optimizer_updates=2000)
            optimizer_old = make_optimizer(config_old)
            optimizer_new = make_optimizer(config_new)
            parent450 = load_parent(
                repository_root, parent_artifact_root, 0, condition, update=450
            )
            parent500 = load_parent(
                repository_root, parent_artifact_root, 0, condition, update=500
            )
            target = make_target(config_old)
            old_step = make_train_step(
                config_old, condition.representation, optimizer_old
            )
            new_step = make_train_step(
                config_new, condition.representation, optimizer_new
            )
            _, sample = sample_training_batch(parent450.data_key, config_old)
            compiled_old = old_step.lower(
                parent450.state, sample, target, parent450.wires
            ).compile()
            compiled_new = new_step.lower(
                parent500.state, sample, target, parent500.wires
            ).compile()

            replay_state, replay_key = parent450.state, parent450.data_key
            for _update in range(451, 501):
                replay_key, inputs = sample_training_batch(replay_key, config_old)
                replay_state, _, _, _ = compiled_old(
                    replay_state, inputs, target, parent450.wires
                )
            jax.block_until_ready(replay_state.params)
            historical = {
                **compare_states(replay_state, parent500.state),
                "data_key": np.array_equal(
                    np.asarray(replay_key), np.asarray(parent500.data_key)
                ),
            }

            old_state, old_key = parent500.state, parent500.data_key
            new_state, new_key = parent500.state, parent500.data_key
            budget_steps = []
            for update in range(501, 505):
                old_key, old_inputs = sample_training_batch(old_key, config_old)
                new_key, new_inputs = sample_training_batch(new_key, config_new)
                old_state, old_loss, _, _ = compiled_old(
                    old_state, old_inputs, target, parent500.wires
                )
                new_state, new_loss, _, _ = compiled_new(
                    new_state, new_inputs, target, parent500.wires
                )
                checks = {
                    "inputs": np.array_equal(
                        np.asarray(old_inputs), np.asarray(new_inputs)
                    ),
                    "loss": np.array_equal(np.asarray(old_loss), np.asarray(new_loss)),
                    **compare_states(old_state, new_state),
                    "data_key": np.array_equal(
                        np.asarray(old_key), np.asarray(new_key)
                    ),
                }
                budget_steps.append({"global_update": update, **checks})

            uninterrupted, uninterrupted_key = parent500.state, parent500.data_key
            interrupted, interrupted_key = parent500.state, parent500.data_key
            uninterrupted_hashes = []
            interrupted_hashes = []
            for _update in range(501, 511):
                uninterrupted_key, inputs = sample_training_batch(
                    uninterrupted_key, config_new
                )
                uninterrupted_hashes.append(batch_sha256(inputs))
                uninterrupted, _, _, _ = compiled_new(
                    uninterrupted, inputs, target, parent500.wires
                )
            for _update in range(501, 506):
                interrupted_key, inputs = sample_training_batch(
                    interrupted_key, config_new
                )
                interrupted_hashes.append(batch_sha256(inputs))
                interrupted, _, _, _ = compiled_new(
                    interrupted, inputs, target, parent500.wires
                )
            checkpoint_path = artifact_directory / (
                f"interruption_{condition.name}_update_0505.npz"
            )
            metadata = checkpoint_metadata(
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                seed=0,
                condition=condition,
                parent=parent500,
            )
            save_checkpoint_atomic(
                checkpoint_path, interrupted, interrupted_key, metadata
            )
            interrupted, interrupted_key, _ = load_checkpoint(
                checkpoint_path, parent500.state, metadata
            )
            state_before_eval = tree_sha256(
                (
                    interrupted.params,
                    interrupted.opt_state,
                    interrupted.key,
                    interrupted.update_index,
                    interrupted_key,
                )
            )
            probe = _probe_path(repository_root, document)
            with np.load(probe, allow_pickle=False) as stored:
                probe_inputs = jnp.asarray(stored["inputs"])
            for hardening in (None, "native", "common"):
                jax.device_get(
                    evaluate_params(
                        probe_inputs,
                        interrupted.params,
                        parent500.wires,
                        condition.representation,
                        config_new,
                        hardening,
                    )
                )
            state_after_eval = tree_sha256(
                (
                    interrupted.params,
                    interrupted.opt_state,
                    interrupted.key,
                    interrupted.update_index,
                    interrupted_key,
                )
            )
            for _update in range(506, 511):
                interrupted_key, inputs = sample_training_batch(
                    interrupted_key, config_new
                )
                interrupted_hashes.append(batch_sha256(inputs))
                interrupted, _, _, _ = compiled_new(
                    interrupted, inputs, target, parent500.wires
                )
            interruption = {
                **compare_states(uninterrupted, interrupted),
                "data_key": np.array_equal(
                    np.asarray(uninterrupted_key), np.asarray(interrupted_key)
                ),
                "batch_hashes": uninterrupted_hashes == interrupted_hashes,
                "evaluation_did_not_mutate_live_state": state_before_eval
                == state_after_eval,
            }
            report["conditions"][condition.name] = {
                "historical_replay_450_to_500": historical,
                "budget_field_parity": budget_steps,
                "interruption_parity": interruption,
                "checkpoint": _artifact(
                    checkpoint_path, "R004_preflight_interruption_checkpoint"
                ),
            }
            write_json(report_path, report)

        common_success = {name: 0 for name in conditions}
        pairings: dict[int, dict[str, Any]] = {}
        boundary_parents = []
        soft_max_abs = 0.0
        soft_replay_failures = []
        boolean_replay_failures = []
        saved_first_probe_failures = []
        for seed in seeds:
            pairings[seed] = {}
            for condition in conditions.values():
                parent = load_parent(
                    repository_root, parent_artifact_root, seed, condition
                )
                boundary_parents.append(parent)
                pairings[seed][condition.name] = {
                    "data_key": np.asarray(parent.data_key).tolist(),
                    "wiring_sha256": parent.manifest["pairing"]["wiring_sha256"],
                    "training_stream_sha256": parent.manifest["pairing"][
                        "training_stream_sha256"
                    ],
                }
                config = config_for(seed, condition, optimizer_updates=2000)
                probe = _probe_path(repository_root, document)
                with np.load(probe, allow_pickle=False) as stored:
                    inputs = jnp.asarray(stored["inputs"])
                    target_array = stored["target"]
                soft, native, common = jax.device_get(
                    jax.jit(
                        lambda params: (
                            evaluate_params(
                                inputs,
                                params,
                                parent.wires,
                                condition.representation,
                                config,
                            ),
                            evaluate_params(
                                inputs,
                                params,
                                parent.wires,
                                condition.representation,
                                config,
                                "native",
                            ),
                            evaluate_params(
                                inputs,
                                params,
                                parent.wires,
                                condition.representation,
                                config,
                                "common",
                            ),
                        )
                    )(parent.state.params)
                )
                soft = np.asarray(soft)
                soft_loss = float(np.square(soft[..., 0] - target_array[..., 0]).sum())
                comparison = _comparison(
                    soft_loss,
                    parent.manifest["evaluations"][-1][
                        "soft_terminal_summed_squared_error"
                    ],
                    1e-6,
                    1e-4,
                )
                soft_max_abs = max(soft_max_abs, comparison["max_abs_error"])
                if not comparison["ok"]:
                    soft_replay_failures.append(
                        {
                            "run_id": parent.manifest["run_id"],
                            "recorded_loss": parent.manifest["evaluations"][-1][
                                "soft_terminal_summed_squared_error"
                            ],
                            "replayed_loss": soft_loss,
                            "comparison": comparison,
                        }
                    )
                parent_evaluation = parent.manifest["evaluations"][-1]
                for hardening, prediction in (("native", native), ("common", common)):
                    replayed = _hard_metrics(np.asarray(prediction), target_array)
                    if replayed != parent_evaluation[hardening]:
                        boolean_replay_failures.append(
                            {
                                "run_id": parent.manifest["run_id"],
                                "hardening": hardening,
                                "recorded": parent_evaluation[hardening],
                                "replayed": replayed,
                            }
                        )
                common_success[condition.name] += int(
                    _hard_metrics(np.asarray(common), target_array)["exact_all_32"]
                )
                trajectory_reference = _resolve_parent_artifact(
                    _record_for_name(
                        parent.manifest, "probe_trajectory_update_500.npz"
                    ),
                    parent_artifact_root,
                    parent.manifest["run_id"],
                )
                with np.load(trajectory_reference, allow_pickle=False) as stored:
                    saved_first_soft = stored["soft"]
                replayed_first_soft = np.asarray(
                    batch_trajectory_q(
                        inputs[:1],
                        effective_truth_tables(
                            parent.state.params, condition.representation
                        ),
                        parent.wires,
                        config.periodic,
                        config.runtime_ticks,
                    )
                )
                first_comparison = _comparison(
                    replayed_first_soft, saved_first_soft, 1e-6, 1e-4
                )
                if not first_comparison["ok"]:
                    saved_first_probe_failures.append(
                        {
                            "run_id": parent.manifest["run_id"],
                            "comparison": first_comparison,
                        }
                    )
            values = list(pairings[seed].values())
            if len({json.dumps(value["data_key"]) for value in values}) != 1:
                raise ValueError(f"R004 seed {seed} parent data keys are not paired")
            if len({value["wiring_sha256"] for value in values}) != 1:
                raise ValueError(f"R004 seed {seed} parent wiring is not paired")
            if len({value["training_stream_sha256"] for value in values}) != 1:
                raise ValueError(f"R004 seed {seed} parent streams are not paired")
        expected_success = dict(zip(conditions, (1, 1, 2, 0)))
        from .r004_analysis import (
            adapt_parent_manifest,
            runtime_record,
            structural_record,
        )

        adapted = [
            adapt_parent_manifest(parent, parent_artifact_root)
            for parent in boundary_parents
        ]
        expected_modes = json.loads(
            (
                repository_root
                / "results/R002_formal_attempt01_runtime_profile_attempt01/runs.json"
            ).read_text()
        )["modes"]
        expected_structures = {
            (item["run_id"], item["hardening"]): {
                key: item["structure"][key]
                for key in (
                    "visible_core_binary_nodes",
                    "visible_recurrent_channels",
                    "binary_nodes_all_channels",
                    "and_nodes",
                    "xor_nodes",
                    "relevant_local_inputs",
                )
            }
            for item in expected_modes
        }
        adapted_structures = [
            structural_record(item, 500, hardening)
            for item in adapted
            for hardening in ("common", "native")
        ]
        structure_exact = all(
            {
                key: item[key]
                for key in (
                    "visible_core_binary_nodes",
                    "visible_recurrent_channels",
                    "binary_nodes_all_channels",
                    "and_nodes",
                    "xor_nodes",
                    "relevant_local_inputs",
                )
            }
            == expected_structures[(item["run_id"], item["hardening"])]
            for item in adapted_structures
        )
        with np.load(
            _probe_path(repository_root, document), allow_pickle=False
        ) as stored:
            original_inputs = stored["inputs"].astype(np.uint8)
            runtime_target = stored["target"][0, ..., 0].astype(np.uint8)
        fresh_path = (
            repository_root / document["runtime_profile"]["additional_probe_path"]
        )
        if (
            sha256_file(fresh_path)
            != document["runtime_profile"]["additional_probe_sha256"]
        ):
            raise ValueError("R004 preflight fresh probe identity mismatch")
        with np.load(fresh_path, allow_pickle=False) as stored:
            fresh_inputs = stored["inputs"].astype(np.uint8)
        from scripts.r002_runtime_profile import make_initializations

        initializations, labels = make_initializations(original_inputs, fresh_inputs)
        selected_runtime = {}
        for run_id, expected_status, expected_tick, phase in (
            (
                "R002_formal_attempt01_seed04_categorical_reference_decay",
                "all_initial_states_visible_target_certified",
                16,
                False,
            ),
            (
                "R002_formal_attempt01_seed14_truth_reference_decay",
                "inconclusive",
                None,
                True,
            ),
        ):
            item = next(value for value in adapted if value["run_id"] == run_id)
            mode, _ = runtime_record(
                item,
                500,
                "common",
                initializations,
                labels,
                runtime_target,
                256,
                [20, 21, 22, 25, 32, 64, 128, 256],
            )
            selected_runtime[run_id] = {
                "universal_status": mode["universal_state_certificate"]["status"],
                "universal_onset": mode["universal_state_certificate"][
                    "certified_visible_settling_tick"
                ],
                "observed_phase_dependence": mode["observed_phase_dependence"],
                "expected": {
                    "universal_status": expected_status,
                    "universal_onset": expected_tick,
                    "observed_phase_dependence": phase,
                },
            }
        analysis_adapter_passed = structure_exact and all(
            {
                key: value[key]
                for key in (
                    "universal_status",
                    "universal_onset",
                    "observed_phase_dependence",
                )
            }
            == value["expected"]
            for value in selected_runtime.values()
        )
        report["cohort_boundary"] = {
            "parents_verified": 64,
            "pairing_verified": True,
            "common_success_counts": common_success,
            "expected_common_success_counts": expected_success,
            "success_counts_exact": common_success == expected_success,
            "maximum_soft_loss_absolute_residual": soft_max_abs,
            "soft_loss_replay_failures": soft_replay_failures,
            "soft_loss_replay_passed": not soft_replay_failures,
            "boolean_replay_failures": boolean_replay_failures,
            "boolean_replay_passed": not boolean_replay_failures,
            "saved_first_probe_soft_trajectory_failures": saved_first_probe_failures,
            "saved_first_probe_soft_trajectories_passed": not saved_first_probe_failures,
        }
        report["analysis_adapter"] = {
            "labeled_structural_modes_verified": len(adapted_structures),
            "structural_counts_exact": structure_exact,
            "selected_runtime_regressions": selected_runtime,
            "passed": analysis_adapter_passed,
        }
        condition_checks = []
        for item in report["conditions"].values():
            condition_checks.extend(item["historical_replay_450_to_500"].values())
            condition_checks.extend(
                value
                for step in item["budget_field_parity"]
                for value in step.values()
                if isinstance(value, bool)
            )
            condition_checks.extend(item["interruption_parity"].values())
        report["status"] = (
            "passed"
            if all(condition_checks)
            and report["cohort_boundary"]["success_counts_exact"]
            and report["cohort_boundary"]["soft_loss_replay_passed"]
            and report["cohort_boundary"]["boolean_replay_passed"]
            and report["cohort_boundary"]["saved_first_probe_soft_trajectories_passed"]
            and report["analysis_adapter"]["passed"]
            else "failed"
        )
        report["ended_at"] = utc_now()
        report["artifact_volume_projection"] = {
            "preflight_checkpoint_bytes": sum(
                item["checkpoint"]["bytes"] for item in report["conditions"].values()
            ),
            "projected_new_checkpoint_count": 1920,
            "method": "condition-specific seed-0 interruption checkpoint size times 480 checkpoints per condition",
            "projected_checkpoint_bytes": sum(
                item["checkpoint"]["bytes"] * 480
                for item in report["conditions"].values()
            ),
        }
        write_json(report_path, report)
    except BaseException as error:
        report["status"] = "failed"
        report["ended_at"] = utc_now()
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        write_json(report_path, report)
        raise
    return report
