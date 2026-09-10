"""R002 gate-coordinate comparison on a common multilinear truth-table kernel."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .difflogic_ca import (
    Params,
    SyncConfig,
    TrainState,
    Wires,
    get_grid_patches,
    init_diff_logic_ca,
    sample_training_batch,
)

Representation = Literal["categorical", "truth"]
Hardening = Literal["native", "common"]

TRUTH_COLUMNS = ("00", "01", "10", "11")
TRUTH_BIT_WEIGHTS = jnp.array([8, 4, 2, 1], dtype=jnp.int32)
GATE_TRUTH_TABLES = jnp.array(
    [[(gate >> (3 - column)) & 1 for column in range(4)] for gate in range(16)],
    dtype=jnp.float32,
)


@dataclass(frozen=True)
class Condition:
    name: str
    representation: Representation
    weight_decay: float


CONDITIONS = {
    condition.name: condition
    for condition in (
        Condition("categorical_reference_decay", "categorical", 0.01),
        Condition("truth_reference_decay", "truth", 0.01),
        Condition("categorical_no_decay", "categorical", 0.0),
        Condition("truth_no_decay", "truth", 0.0),
    )
}


class PairedInitialization(NamedTuple):
    categorical_params: Params
    truth_params: Params
    wires: Wires
    model_key: jax.Array


def config_for(
    seed: int, condition: Condition, *, optimizer_updates: int = 500
) -> SyncConfig:
    return dataclasses.replace(
        SyncConfig(),
        seed=seed,
        weight_decay=condition.weight_decay,
        optimizer_updates=optimizer_updates,
    )


def make_optimizer(config: SyncConfig) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip(config.clip_value),
        optax.adamw(
            learning_rate=config.learning_rate,
            b1=config.adam_b1,
            b2=config.adam_b2,
            weight_decay=config.weight_decay,
        ),
    )


def effective_truth_tables(params: Params, representation: Representation) -> Params:
    if representation == "categorical":
        return jax.tree_util.tree_map(
            lambda logits: jnp.matmul(
                jax.nn.softmax(logits, axis=-1),
                GATE_TRUTH_TABLES,
                precision=jax.lax.Precision.HIGHEST,
            ),
            params,
        )
    if representation == "truth":
        return jax.tree_util.tree_map(jax.nn.sigmoid, params)
    raise ValueError(f"unsupported representation: {representation}")


def categorical_to_truth_params(categorical_params: Params) -> Params:
    """Map source logits to logit(p @ T), with no clamping."""
    q = effective_truth_tables(categorical_params, "categorical")
    return jax.tree_util.tree_map(lambda value: jnp.log(value) - jnp.log1p(-value), q)


def init_paired(seed: int) -> PairedInitialization:
    config = dataclasses.replace(SyncConfig(), seed=seed)
    key = jax.random.PRNGKey(seed)
    model_key, init_key = jax.random.split(key)
    categorical, wires = init_diff_logic_ca(config, init_key)
    truth = categorical_to_truth_params(categorical)
    return PairedInitialization(categorical, truth, wires, model_key)


def init_condition_state(
    seed: int, condition: Condition, optimizer: optax.GradientTransformation
) -> tuple[TrainState, Wires, PairedInitialization]:
    paired = init_paired(seed)
    params = (
        paired.categorical_params
        if condition.representation == "categorical"
        else paired.truth_params
    )
    return (
        TrainState(params, optimizer.init(params), paired.model_key, jnp.array(0)),
        paired.wires,
        paired,
    )


def multilinear_gate(a: jax.Array, b: jax.Array, q: jax.Array) -> jax.Array:
    basis = jnp.stack(((1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b), axis=-1)
    return jnp.sum(basis * q[None, ...], axis=-1)


def gate_ids(
    params: Params, representation: Representation, hardening: Hardening
) -> Params:
    if hardening == "native" and representation == "categorical":
        return jax.tree_util.tree_map(lambda value: jnp.argmax(value, axis=-1), params)
    q = effective_truth_tables(params, representation)
    return jax.tree_util.tree_map(
        lambda value: jnp.sum(
            (value >= jnp.float32(0.5)).astype(jnp.int32) * TRUTH_BIT_WEIGHTS,
            axis=-1,
        ),
        q,
    )


def hardened_truth_tables(
    params: Params, representation: Representation, hardening: Hardening
) -> Params:
    ids = gate_ids(params, representation, hardening)
    return jax.tree_util.tree_map(lambda value: GATE_TRUTH_TABLES[value], ids)


def _run_layer_q(q: jax.Array, wires: list[jax.Array], x: jax.Array) -> jax.Array:
    return multilinear_gate(x[..., wires[0]], x[..., wires[1]], q)


def _run_perceive_q(q_params, wires, patch):
    run_layer_map = jax.vmap(_run_layer_q, in_axes=(0, None, 0))
    previous = patch
    x = jnp.transpose(patch, (1, 0))
    x = jnp.repeat(x[None, ...], q_params[0].shape[0], axis=0)
    for q, layer_wires in zip(q_params, wires):
        x = run_layer_map(q, layer_wires, x)
    x = jnp.transpose(x, (1, 2, 0)).reshape(-1)
    return jnp.concatenate([previous[4, :], x], axis=-1)


def _run_circuit_q(q_params: Params, wires: Wires, patch: jax.Array) -> jax.Array:
    x = _run_perceive_q(q_params["perceive"], wires["perceive"], patch)
    for q, layer_wires in zip(q_params["update"], wires["update"]):
        x = _run_layer_q(q, layer_wires, x)
    return x


def run_sync_q(
    grid: jax.Array, q_params: Params, wires: Wires, periodic: bool
) -> jax.Array:
    patches = get_grid_patches(grid, 3, grid.shape[-1], periodic)
    updated = jax.vmap(_run_circuit_q, in_axes=(None, None, 0))(
        q_params, wires, patches
    )
    return updated.reshape(grid.shape)


def rollout_q(
    grid: jax.Array,
    q_params: Params,
    wires: Wires,
    periodic: bool,
    runtime_ticks: int,
) -> jax.Array:
    def body(current, _unused):
        return run_sync_q(current, q_params, wires, periodic), None

    final, _ = jax.lax.scan(body, grid, None, length=runtime_ticks)
    return final


def rollout_trajectory_q(
    grid: jax.Array,
    q_params: Params,
    wires: Wires,
    periodic: bool,
    runtime_ticks: int,
) -> jax.Array:
    def body(current, _unused):
        updated = run_sync_q(current, q_params, wires, periodic)
        return updated, updated

    _, states = jax.lax.scan(body, grid, None, length=runtime_ticks)
    return jnp.concatenate([grid[None, ...], states], axis=0)


def batch_rollout_q(
    grids: jax.Array,
    q_params: Params,
    wires: Wires,
    periodic: bool,
    runtime_ticks: int,
) -> jax.Array:
    return jax.vmap(rollout_q, in_axes=(0, None, None, None, None))(
        grids, q_params, wires, periodic, runtime_ticks
    )


def batch_trajectory_q(
    grids: jax.Array,
    q_params: Params,
    wires: Wires,
    periodic: bool,
    runtime_ticks: int,
) -> jax.Array:
    trajectories = jax.vmap(rollout_trajectory_q, in_axes=(0, None, None, None, None))(
        grids, q_params, wires, periodic, runtime_ticks
    )
    return jnp.swapaxes(trajectories, 0, 1)


def evaluate_params(
    inputs: jax.Array,
    params: Params,
    wires: Wires,
    representation: Representation,
    config: SyncConfig,
    hardening: Hardening | None = None,
) -> jax.Array:
    q = (
        effective_truth_tables(params, representation)
        if hardening is None
        else hardened_truth_tables(params, representation, hardening)
    )
    return batch_rollout_q(inputs, q, wires, config.periodic, config.runtime_ticks)


def loss(
    params: Params,
    wires: Wires,
    inputs: jax.Array,
    target: jax.Array,
    representation: Representation,
    config: SyncConfig,
) -> jax.Array:
    predicted = evaluate_params(inputs, params, wires, representation, config)
    return jnp.square(predicted[..., 0] - target[..., 0]).sum()


def make_train_step(
    config: SyncConfig,
    representation: Representation,
    optimizer: optax.GradientTransformation,
):
    value_and_grad = jax.value_and_grad(
        lambda params, wires, inputs, target: loss(
            params, wires, inputs, target, representation, config
        )
    )

    @jax.jit
    def train_step(
        state: TrainState, inputs: jax.Array, target: jax.Array, wires: Wires
    ):
        key, _unused_loss_key = jax.random.split(state.key)
        value, gradients = value_and_grad(state.params, wires, inputs, target)
        updates, opt_state = optimizer.update(gradients, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        return (
            TrainState(params, opt_state, key, state.update_index + 1),
            value,
            gradients,
            updates,
        )

    return train_step


def tree_sha256(tree: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        value = np.asarray(leaf)
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def ordered_training_stream(seed: int, config: SyncConfig, updates: int):
    key = jax.random.PRNGKey(seed)
    batches = []
    for _ in range(updates):
        key, batch = sample_training_batch(key, config)
        batches.append(batch)
    return key, batches


def stream_sha256(batches: list[jax.Array]) -> str:
    return tree_sha256(batches)


def q_displacement(
    initial_q: Params, params: Params, representation: Representation
) -> dict[str, float]:
    current = effective_truth_tables(params, representation)
    differences = [
        value - initial
        for value, initial in zip(
            jax.tree_util.tree_leaves(current), jax.tree_util.tree_leaves(initial_q)
        )
    ]
    return {
        "max_abs": float(max(jnp.max(jnp.abs(value)) for value in differences)),
        "l2": float(jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in differences))),
    }


def validate_config_contract(
    document: dict[str, Any],
) -> tuple[tuple[int, ...], dict[str, Condition]]:
    if document.get("experiment_id") != "R002":
        raise ValueError("R002 config must declare experiment_id=R002")
    recipe = document["scientific_recipe"]
    reference = SyncConfig()
    expected = {
        "seeds": list(range(16)),
        "optimizer_updates": 500,
        "runtime_ticks": reference.runtime_ticks,
        "batch_size": reference.batch_size,
        "grid_size": list(reference.grid_size),
        "channels": reference.channels,
        "learning_rate": reference.learning_rate,
        "adam_b1": reference.adam_b1,
        "adam_b2": reference.adam_b2,
        "clip_value": reference.clip_value,
        "loss": "summed_terminal_channel_0_squared_error",
        "periodic": False,
        "kernel": "common_multilinear_truth_table",
        "truth_column_order": list(TRUTH_COLUMNS),
        "common_rounding": "q>=0.5",
    }
    mismatches = {
        name: (recipe.get(name), value)
        for name, value in expected.items()
        if recipe.get(name) != value
    }
    declared = {
        item["name"]: Condition(
            item["name"], item["representation"], item["weight_decay"]
        )
        for item in document["conditions"]
    }
    if declared != CONDITIONS:
        mismatches["conditions"] = (declared, CONDITIONS)
    if mismatches:
        raise ValueError(f"R002 config differs from frozen recipe: {mismatches}")
    return tuple(recipe["seeds"]), declared


def save_checkpoint(
    path: Path,
    state: TrainState,
    data_key: jax.Array,
    metadata: dict[str, Any],
) -> None:
    params, _ = jax.tree_util.tree_flatten(state.params)
    optimizer, _ = jax.tree_util.tree_flatten(state.opt_state)
    arrays: dict[str, Any] = {
        "format_version": np.array(2),
        "update_index": np.asarray(state.update_index),
        "model_key": np.asarray(state.key),
        "data_key": np.asarray(data_key),
        "param_leaf_count": np.array(len(params)),
        "opt_leaf_count": np.array(len(optimizer)),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    arrays.update({f"param_{i:03d}": np.asarray(v) for i, v in enumerate(params)})
    arrays.update({f"opt_{i:03d}": np.asarray(v) for i, v in enumerate(optimizer)})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_checkpoint(
    path: Path,
    template: TrainState,
    expected_metadata: dict[str, Any],
) -> tuple[TrainState, jax.Array, dict[str, Any]]:
    param_leaves, param_tree = jax.tree_util.tree_flatten(template.params)
    opt_leaves, opt_tree = jax.tree_util.tree_flatten(template.opt_state)
    with np.load(path, allow_pickle=False) as stored:
        if int(stored["format_version"]) != 2:
            raise ValueError("unsupported R002 checkpoint format")
        metadata = json.loads(str(stored["metadata_json"]))
        for name, value in expected_metadata.items():
            if metadata.get(name) != value:
                raise ValueError(
                    f"checkpoint metadata mismatch for {name}: {metadata.get(name)!r} != {value!r}"
                )
        if int(stored["param_leaf_count"]) != len(param_leaves):
            raise ValueError(
                "checkpoint parameter representation does not match template"
            )
        if int(stored["opt_leaf_count"]) != len(opt_leaves):
            raise ValueError("checkpoint optimizer does not match template")
        state = TrainState(
            jax.tree_util.tree_unflatten(
                param_tree,
                [
                    jnp.asarray(stored[f"param_{i:03d}"])
                    for i in range(len(param_leaves))
                ],
            ),
            jax.tree_util.tree_unflatten(
                opt_tree,
                [jnp.asarray(stored[f"opt_{i:03d}"]) for i in range(len(opt_leaves))],
            ),
            jnp.asarray(stored["model_key"]),
            jnp.asarray(stored["update_index"]),
        )
        return state, jnp.asarray(stored["data_key"]), metadata
