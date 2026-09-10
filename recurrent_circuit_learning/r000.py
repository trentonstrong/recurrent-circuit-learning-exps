"""R000 fixture, parity, checkpoint, and profiling operations."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import optax

from .checkpoint import load_checkpoint, save_checkpoint
from .difflogic_ca import (
    SyncConfig,
    TrainState,
    batch_trajectory,
    bin_op_all_combinations,
    flatten_float_tree,
    get_grid_patches,
    init_train_state,
    loss,
    make_optimizer,
    make_target,
    make_train_step,
    nontrivial_fixed_params,
    replace_params,
    sample_training_batch,
)
from .notebook_oracle import load_notebook_namespace

SOURCE_SHA256 = "a9b3829db0d9fe0eb46148d516c18358aa648e3538aeaa682003b77cfbe757c8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != SOURCE_SHA256:
        raise ValueError(f"source checksum mismatch: {actual} != {SOURCE_SHA256}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def environment_metadata() -> dict[str, Any]:
    signature = inspect.signature(optax.adamw)
    devices = jax.devices()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "optax": optax.__version__,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in devices],
        "optax_adamw_defaults": {
            name: str(signature.parameters[name].default)
            for name in ("eps", "eps_root", "mu_dtype", "nesterov")
            if name in signature.parameters
        },
    }


def _wire_arrays(wires: dict[str, list[list[jax.Array]]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for network_name in ("perceive", "update"):
        for layer_index, (wire_a, wire_b) in enumerate(wires[network_name]):
            arrays[f"wire_{network_name}_{layer_index:02d}_a"] = np.asarray(wire_a)
            arrays[f"wire_{network_name}_{layer_index:02d}_b"] = np.asarray(wire_b)
    return arrays


def _parameter_results(
    label: str,
    state,
    wires,
    inputs,
    target,
    config,
    optimizer,
) -> dict[str, np.ndarray]:
    train_step = make_train_step(config, optimizer)
    trajectory = jax.jit(
        lambda params: batch_trajectory(
            inputs,
            params,
            wires,
            True,
            config.periodic,
            config.runtime_ticks,
        )
    )(state.params)
    hard_trajectory = jax.jit(
        lambda params: batch_trajectory(
            inputs,
            params,
            wires,
            False,
            config.periodic,
            config.runtime_ticks,
        )
    )(state.params)
    value, gradients = jax.jit(
        jax.value_and_grad(
            lambda params: loss(params, wires, inputs, target, config, True)
        )
    )(state.params)
    next_state, step_loss, auxiliary, step_gradients = train_step(
        state, inputs, target, wires
    )
    jax.block_until_ready(next_state.params)
    return {
        f"{label}_params": flatten_float_tree(state.params),
        f"{label}_soft_trajectory": np.asarray(trajectory),
        f"{label}_hard_trajectory": np.asarray(hard_trajectory),
        f"{label}_loss": np.asarray(value),
        f"{label}_gradients": flatten_float_tree(gradients),
        f"{label}_step_loss": np.asarray(step_loss),
        f"{label}_step_hard_loss": np.asarray(auxiliary["hard"]),
        f"{label}_step_gradients": flatten_float_tree(step_gradients),
        f"{label}_updated_params": flatten_float_tree(next_state.params),
        f"{label}_opt_state": flatten_float_tree(state.opt_state),
        f"{label}_updated_opt_state": flatten_float_tree(next_state.opt_state),
        f"{label}_updated_model_key": np.asarray(next_state.key),
    }


def _gate_id_arrays(params: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for network in ("perceive", "update"):
        for index, logits in enumerate(params[network]):
            arrays[f"hard_gate_id_{network}_{index:02d}"] = np.asarray(
                jnp.argmax(logits, axis=-1)
            )
    return arrays


def _sample_batches(key: jax.Array, config: SyncConfig, count: int = 3):
    arrays: dict[str, np.ndarray] = {"data_key_before": np.asarray(key)}
    for index in range(count):
        key, inputs = sample_training_batch(key, config)
        arrays[f"training_inputs_{index:03d}"] = np.asarray(inputs)
        arrays[f"data_key_after_{index:03d}"] = np.asarray(key)
    # Retain the schema-v1 names for the first batch.
    arrays["training_inputs"] = arrays["training_inputs_000"]
    arrays["data_key_after"] = arrays["data_key_after_000"]
    return key, arrays


def build_fixture() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = SyncConfig()
    optimizer = make_optimizer(config)
    initial_state, wires = init_train_state(config, optimizer)
    data_key_before = jax.random.PRNGKey(config.seed)
    _, sampled = _sample_batches(data_key_before, config)
    inputs = jnp.asarray(sampled["training_inputs"])
    target = make_target(config)

    boolean_pairs = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
    fractions = jnp.array([0.0, 0.125, 0.5, 0.875, 1.0], dtype=jnp.float64)
    fractional_pairs = jnp.stack(jnp.meshgrid(fractions, fractions), axis=-1).reshape(
        -1, 2
    )
    patch_grid = jnp.arange(1, 13, dtype=jnp.float32).reshape(2, 3, 2)

    arrays: dict[str, np.ndarray] = {
        "boolean_pairs": np.asarray(boolean_pairs),
        "boolean_gate_values": np.asarray(
            bin_op_all_combinations(boolean_pairs[:, 0], boolean_pairs[:, 1])
        ),
        "fractional_pairs_fp64": np.asarray(fractional_pairs),
        "fractional_gate_values_fp64": np.asarray(
            bin_op_all_combinations(fractional_pairs[:, 0], fractional_pairs[:, 1])
        ),
        "patch_grid": np.asarray(patch_grid),
        "zero_boundary_patches": np.asarray(get_grid_patches(patch_grid, 3, 2, False)),
        "target": np.asarray(target),
        "model_key": np.asarray(initial_state.key),
    }
    arrays.update(sampled)
    arrays.update(_wire_arrays(wires))
    arrays.update(_gate_id_arrays(initial_state.params))
    arrays.update(
        _parameter_results(
            "identity", initial_state, wires, inputs, target, config, optimizer
        )
    )
    nontrivial_state = replace_params(
        initial_state, nontrivial_fixed_params(initial_state.params)
    )
    arrays.update(
        _parameter_results(
            "nontrivial", nontrivial_state, wires, inputs, target, config, optimizer
        )
    )
    metadata = {
        "schema_version": 1,
        "source_sha256": SOURCE_SHA256,
        "environment": environment_metadata(),
        "scientific_dtype": "float32",
        "fp64_probe_enabled": bool(jax.config.x64_enabled),
        "gate_truth_column_order": ["00", "01", "10", "11"],
        "adaptations": [
            "flax.linen.softmax replaced by equivalent jax.nn.softmax",
            "einops rearrange k c s -> (c s k) expressed as transpose plus reshape",
            "unused trailing connection labels omitted from standalone configuration",
            "jax_threefry_partitionable fixed false to retain JAX 0.4.33 seeded arrays",
        ],
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    return arrays, metadata


def _notebook_trajectory(namespace: dict[str, Any], inputs, params, wires, training):
    def one_grid(grid):
        def body(current, _unused):
            updated = namespace["run_sync"](current, params, wires, training, 0)
            return updated, updated

        _, states = jax.lax.scan(body, grid, None, length=20)
        return jnp.concatenate([grid[None, ...], states], axis=0)

    return jnp.swapaxes(jax.vmap(one_grid)(inputs), 0, 1)


def _notebook_parameter_results(
    namespace: dict[str, Any],
    label: str,
    state: Any,
    wires: Any,
    inputs: Any,
    target: Any,
) -> dict[str, np.ndarray]:
    trajectory = jax.jit(
        lambda params: _notebook_trajectory(namespace, inputs, params, wires, 1)
    )(state.param)
    hard_trajectory = jax.jit(
        lambda params: _notebook_trajectory(namespace, inputs, params, wires, 0)
    )(state.param)
    value, gradients = jax.jit(
        jax.value_and_grad(
            lambda params: namespace["loss_f"](
                params, wires, inputs, target, 0, 20, False, state.key
            )[0]
        )
    )(state.param)
    next_state, step_loss, auxiliary = namespace["train_step"](
        state, inputs, target, wires, 0, 20, False
    )
    _, step_gradients = jax.value_and_grad(
        lambda params: namespace["loss_f"](
            params, wires, inputs, target, 0, 20, False, jax.random.split(state.key)[1]
        )[0]
    )(state.param)
    jax.block_until_ready(next_state.param)
    return {
        f"{label}_params": flatten_float_tree(state.param),
        f"{label}_soft_trajectory": np.asarray(trajectory),
        f"{label}_hard_trajectory": np.asarray(hard_trajectory),
        f"{label}_loss": np.asarray(value),
        f"{label}_gradients": flatten_float_tree(gradients),
        f"{label}_step_loss": np.asarray(step_loss),
        f"{label}_step_hard_loss": np.asarray(auxiliary["hard"]),
        f"{label}_step_gradients": flatten_float_tree(step_gradients),
        f"{label}_updated_params": flatten_float_tree(next_state.param),
        f"{label}_opt_state": flatten_float_tree(state.opt_state),
        f"{label}_updated_opt_state": flatten_float_tree(next_state.opt_state),
        f"{label}_updated_model_key": np.asarray(next_state.key),
    }


def build_notebook_fixture(
    notebook_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build an X64-disabled fixture from executed vendored-notebook cells."""
    if jax.config.x64_enabled:
        raise RuntimeError("notebook sampling fixture must run with JAX X64 disabled")
    namespace, execution = load_notebook_namespace(notebook_path)
    hyperparams = namespace["hyperparams"]
    state, wires = namespace["init_state"](
        hyperparams, namespace["opt"], hyperparams["seed"]
    )
    config = SyncConfig()
    data_key = jax.random.PRNGKey(config.seed)
    _, sampled = _sample_batches(data_key, config)
    inputs = jnp.asarray(sampled["training_inputs"])
    target = make_target(config)
    pairs = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
    patch_grid = jnp.arange(1, 13, dtype=jnp.float32).reshape(2, 3, 2)
    arrays: dict[str, np.ndarray] = {
        "boolean_pairs": np.asarray(pairs),
        "boolean_gate_values": np.asarray(
            namespace["bin_op_all_combinations"](pairs[:, 0], pairs[:, 1])
        ),
        "patch_grid": np.asarray(patch_grid),
        "zero_boundary_patches": np.asarray(
            namespace["get_grid_patches"](patch_grid, 3, 2, 0)
        ),
        "target": np.asarray(target),
        "model_key": np.asarray(state.key),
    }
    arrays.update(sampled)
    arrays.update(_wire_arrays(wires))
    arrays.update(_gate_id_arrays(state.param))
    arrays.update(
        _notebook_parameter_results(namespace, "identity", state, wires, inputs, target)
    )
    fixed_params = nontrivial_fixed_params(state.param)
    fixed_state = namespace["TrainState"](
        fixed_params, namespace["opt"].init(fixed_params), state.key
    )
    arrays.update(
        _notebook_parameter_results(
            namespace, "nontrivial", fixed_state, wires, inputs, target
        )
    )
    metadata = {
        "schema_version": 2,
        "kind": "vendored_notebook_execution_oracle",
        "source_sha256": SOURCE_SHA256,
        "environment": environment_metadata(),
        "scientific_dtype": "float32",
        "sampling_x64_enabled": False,
        "execution": execution,
    }
    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    return arrays, metadata


def export_notebook_fixture(notebook_path: Path, fixture_path: Path) -> dict[str, Any]:
    arrays, metadata = build_notebook_fixture(notebook_path)
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fixture_path, **arrays)
    return {
        "fixture": str(fixture_path),
        "bytes": fixture_path.stat().st_size,
        "sha256": sha256_file(fixture_path),
        "metadata": metadata,
    }


def _tree_from_flat(template: Any, flat_values: np.ndarray) -> Any:
    leaves, tree = jax.tree_util.tree_flatten(template)
    rebuilt: list[jax.Array] = []
    offset = 0
    for leaf in leaves:
        size = leaf.size
        rebuilt.append(
            jnp.asarray(flat_values[offset : offset + size]).reshape(leaf.shape)
        )
        offset += size
    if offset != flat_values.size:
        raise ValueError(
            f"flat parameter size mismatch: consumed {offset}, got {flat_values.size}"
        )
    return jax.tree_util.tree_unflatten(tree, rebuilt)


def _reference_wires(
    template: dict[str, list[list[jax.Array]]], arrays: dict[str, np.ndarray]
) -> dict[str, list[list[jax.Array]]]:
    wires: dict[str, list[list[jax.Array]]] = {"perceive": [], "update": []}
    for network_name in ("perceive", "update"):
        for layer_index in range(len(template[network_name])):
            wires[network_name].append(
                [
                    jnp.asarray(arrays[f"wire_{network_name}_{layer_index:02d}_a"]),
                    jnp.asarray(arrays[f"wire_{network_name}_{layer_index:02d}_b"]),
                ]
            )
    return wires


def build_candidate_from_reference(
    reference: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Evaluate modern JAX with oracle arrays instead of regenerating its seed."""
    config = SyncConfig()
    optimizer = make_optimizer(config)
    template_state, template_wires = init_train_state(config, optimizer)
    wires = _reference_wires(template_wires, reference)
    inputs = jnp.asarray(reference["training_inputs"])
    target = jnp.asarray(reference["target"])
    model_key = jnp.asarray(reference["model_key"])

    boolean_pairs = jnp.asarray(reference["boolean_pairs"])
    patch_grid = jnp.asarray(reference["patch_grid"])
    arrays: dict[str, np.ndarray] = {
        "boolean_pairs": np.asarray(boolean_pairs),
        "boolean_gate_values": np.asarray(
            bin_op_all_combinations(boolean_pairs[:, 0], boolean_pairs[:, 1])
        ),
        "patch_grid": np.asarray(patch_grid),
        "zero_boundary_patches": np.asarray(get_grid_patches(patch_grid, 3, 2, False)),
        "target": np.asarray(target),
        "model_key": np.asarray(model_key),
    }
    if "fractional_pairs_fp64" in reference:
        fractional_pairs = jnp.asarray(reference["fractional_pairs_fp64"])
        arrays["fractional_pairs_fp64"] = np.asarray(fractional_pairs)
        arrays["fractional_gate_values_fp64"] = np.asarray(
            bin_op_all_combinations(fractional_pairs[:, 0], fractional_pairs[:, 1])
        )
    _, regenerated_samples = _sample_batches(jax.random.PRNGKey(config.seed), config)
    arrays.update(regenerated_samples)
    arrays.update(_wire_arrays(template_wires))
    for label in ("identity", "nontrivial"):
        params = _tree_from_flat(template_state.params, reference[f"{label}_params"])
        state = TrainState(params, optimizer.init(params), model_key, jnp.array(0))
        if label == "identity":
            arrays.update(_gate_id_arrays(params))
        arrays.update(
            _parameter_results(label, state, wires, inputs, target, config, optimizer)
        )

    regenerated_state, regenerated_wires = init_train_state(config, optimizer)
    regenerated_data_key, regenerated_inputs = sample_training_batch(
        jax.random.PRNGKey(config.seed), config
    )
    rng_regeneration = {
        "model_key_matches": bool(
            np.array_equal(np.asarray(regenerated_state.key), reference["model_key"])
        ),
        "data_key_matches": bool(
            np.array_equal(
                np.asarray(regenerated_data_key), reference["data_key_after"]
            )
        ),
        "training_inputs_match": bool(
            np.array_equal(np.asarray(regenerated_inputs), reference["training_inputs"])
        ),
        "successive_training_batches_match": all(
            np.array_equal(value, reference[name])
            for name, value in regenerated_samples.items()
            if name in reference
        ),
        "initial_params_match": bool(
            np.array_equal(
                flatten_float_tree(regenerated_state.params),
                reference["identity_params"],
            )
        ),
        "wires_match": all(
            np.array_equal(value, reference[name])
            for name, value in _wire_arrays(regenerated_wires).items()
        ),
        "interpretation": (
            "Informational only: modern JAX seed regeneration may differ. "
            "Parity uses the CPU oracle's saved keys, inputs, parameters, and wiring."
        ),
    }
    metadata = {
        "schema_version": 1,
        "source_sha256": SOURCE_SHA256,
        "environment": environment_metadata(),
        "scientific_dtype": "float32",
        "fp64_probe_enabled": bool(jax.config.x64_enabled),
        "fixture_input_policy": "load identical CPU-oracle arrays",
    }
    return arrays, metadata, rng_regeneration


def export_fixture(path: Path) -> dict[str, Any]:
    arrays, metadata = build_fixture()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "fixture": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "metadata": metadata,
    }


def _comparison(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    exact: bool,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return {
            "ok": False,
            "actual_shape": actual.shape,
            "expected_shape": expected.shape,
        }
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    max_error = float(difference.max(initial=0.0))
    ok = (
        bool(np.array_equal(actual, expected))
        if exact
        else bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    )
    result: dict[str, Any] = {
        "ok": ok,
        "exact": exact,
        "atol": 0.0 if exact else atol,
        "rtol": 0.0 if exact else rtol,
        "max_abs_error": max_error,
    }
    return result


def verify_fixture(path: Path, report_path: Path | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as stored:
        expected_arrays = {name: stored[name] for name in stored.files}
        reference_metadata = json.loads(str(stored["metadata_json"]))
    actual_arrays, actual_metadata, rng_regeneration = build_candidate_from_reference(
        expected_arrays
    )

    exact_names = {
        "boolean_pairs",
        "boolean_gate_values",
        "patch_grid",
        "zero_boundary_patches",
        "training_inputs",
        "target",
        "model_key",
        "data_key_before",
        "data_key_after",
        "identity_updated_model_key",
        "nontrivial_updated_model_key",
        "identity_hard_trajectory",
        "nontrivial_hard_trajectory",
    }
    exact_names.update(
        name
        for name in expected_arrays
        if name.startswith(("wire_", "hard_gate_id_", "training_inputs_", "data_key_"))
    )
    results: dict[str, Any] = {}
    for name, expected in expected_arrays.items():
        if name == "metadata_json":
            continue
        if name not in actual_arrays:
            results[name] = {"ok": False, "missing": True}
            continue
        is_fp64 = "fp64" in name
        atol, rtol = (1e-10, 1e-8) if is_fp64 else (1e-6, 1e-4)
        # First-step Adam normalization is deliberately sensitive when the
        # identity-biased model has gradients near eps. The observed modern
        # CPU and GPU update errors were 1.34e-5 and 3.05e-5 respectively,
        # while their losses and gradients passed the base FP32 contract.
        if name == "identity_updated_params":
            atol = 5e-5
        results[name] = _comparison(
            actual_arrays[name],
            expected,
            exact=name in exact_names,
            atol=atol,
            rtol=rtol,
        )
        if "trajectory" in name and actual_arrays[name].shape == expected.shape:
            tick_axes = tuple(range(1, expected.ndim))
            tick_error = np.max(
                np.abs(
                    actual_arrays[name].astype(np.float64) - expected.astype(np.float64)
                ),
                axis=tick_axes,
            )
            failing = np.flatnonzero(
                tick_error > (atol + rtol * np.max(np.abs(expected), axis=tick_axes))
            )
            nonzero = np.flatnonzero(tick_error > 0)
            results[name]["earliest_nonzero_tick"] = (
                int(nonzero[0]) if nonzero.size else None
            )
            results[name]["earliest_differing_tick"] = (
                int(failing[0]) if failing.size else None
            )
    report = {
        "schema_version": 1,
        "status": "passed"
        if all(item["ok"] for item in results.values())
        else "failed",
        "reference": reference_metadata,
        "candidate": actual_metadata,
        "seed_regeneration": rng_regeneration,
        "tolerance_notes": {
            "fp64": {"atol": 1e-10, "rtol": 1e-8},
            "fp32": {"atol": 1e-6, "rtol": 1e-4},
            "identity_updated_params": {
                "atol": 5e-5,
                "rtol": 1e-4,
                "reason": (
                    "Measured first-step Adam normalization amplification at "
                    "near-zero identity-biased gradients; losses and gradients "
                    "remain under the base FP32 contract."
                ),
            },
        },
        "comparisons": results,
    }
    if reference_metadata.get("kind") == "vendored_notebook_execution_oracle":
        report["direct_notebook_contract"] = {
            "oracle_does_not_import_extraction": True,
            "all_initialized_arrays_regenerated": all(rng_regeneration.values()),
            "sampling_x64_enabled": reference_metadata["sampling_x64_enabled"],
        }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "passed":
        failed = [name for name, item in results.items() if not item["ok"]]
        raise AssertionError(f"fixture parity failed: {failed}")
    return report


def checkpoint_resume_check(path: Path) -> dict[str, Any]:
    config = SyncConfig()
    optimizer = make_optimizer(config)
    train_step = make_train_step(config, optimizer)
    state, wires = init_train_state(config, optimizer)
    target = make_target(config)
    data_key = jax.random.PRNGKey(config.seed)

    data_key, first_inputs = sample_training_batch(data_key, config)
    first_state, _, _, _ = train_step(state, first_inputs, target, wires)
    jax.block_until_ready(first_state.params)
    save_checkpoint(path, first_state, data_key, {"purpose": "R000 resume check"})

    uninterrupted_data_key, second_inputs = sample_training_batch(data_key, config)
    uninterrupted, _, _, _ = train_step(first_state, second_inputs, target, wires)
    restored, restored_data_key, metadata = load_checkpoint(path, config, optimizer)
    resumed_data_key, resumed_inputs = sample_training_batch(restored_data_key, config)
    resumed, _, _, _ = train_step(restored, resumed_inputs, target, wires)
    jax.block_until_ready((uninterrupted.params, resumed.params))

    checks = {
        "metadata": metadata == {"purpose": "R000 resume check"},
        "input": np.array_equal(np.asarray(second_inputs), np.asarray(resumed_inputs)),
        "data_key": np.array_equal(
            np.asarray(uninterrupted_data_key), np.asarray(resumed_data_key)
        ),
        "model_key": np.array_equal(
            np.asarray(uninterrupted.key), np.asarray(resumed.key)
        ),
        "update_index": int(uninterrupted.update_index)
        == int(resumed.update_index)
        == 2,
        "params": np.array_equal(
            flatten_float_tree(uninterrupted.params), flatten_float_tree(resumed.params)
        ),
        "optimizer": np.array_equal(
            flatten_float_tree(uninterrupted.opt_state),
            flatten_float_tree(resumed.opt_state),
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"checkpoint resume mismatch: {checks}")
    return {"status": "passed", "checks": checks, "checkpoint": str(path)}


def profile_update(repetitions: int = 3) -> dict[str, Any]:
    config = SyncConfig()
    optimizer = make_optimizer(config)
    train_step = make_train_step(config, optimizer)
    state, wires = init_train_state(config, optimizer)
    _, inputs = sample_training_batch(jax.random.PRNGKey(config.seed), config)
    target = make_target(config)

    compile_start = time.perf_counter()
    compiled = train_step.lower(state, inputs, target, wires).compile()
    compile_seconds = time.perf_counter() - compile_start

    run_state = state
    durations: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        run_state, _, _, _ = compiled(run_state, inputs, target, wires)
        jax.block_until_ready(run_state.params)
        durations.append(time.perf_counter() - start)

    device = jax.devices()[0]
    memory_stats = device.memory_stats() or {}
    selected_memory = {
        key: int(value)
        for key, value in memory_stats.items()
        if key in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"}
        and isinstance(value, (int, np.integer))
    }
    return {
        "status": "passed",
        "environment": environment_metadata(),
        "compile_seconds": compile_seconds,
        "synchronized_update_seconds": durations,
        "median_update_seconds": float(np.median(durations)),
        "device_memory": selected_memory,
        "profile_state_discarded": True,
    }
