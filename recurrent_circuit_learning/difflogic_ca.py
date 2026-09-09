"""Standalone JAX implementation of the synchronous DiffLogic CA recipe.

Adapted from ``diffLogic_CA.ipynb`` in Google Research's
``self-organising-systems`` repository at commit
3d5547ca48b60ecac459834e2c05c9ff5df87991.

Copyright 2025 Google LLC. Licensed under the Apache License, Version 2.0.
See ``references/difflogic_ca/LICENSE.apache-2.0.txt`` and the source manifest.

The gate algebra, wiring construction, PRNG splitting, patch order, flattening,
loss reduction, clipping, and AdamW call mirror the executable notebook cells.
Only the synchronous checkerboard path is included here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

# JAX 0.10 defaults this to true, which changes split/permutation outputs for
# the same legacy PRNG key. The pinned 0.4.33 notebook ran with false.
jax.config.update("jax_threefry_partitionable", False)


NUMBER_OF_GATES = 16
PASS_THROUGH_GATE = 3
DEFAULT_PASS_VALUE = 10.0


@dataclass(frozen=True)
class SyncConfig:
    seed: int = 23
    learning_rate: float = 0.05
    batch_size: int = 2
    optimizer_updates: int = 500
    runtime_ticks: int = 20
    channels: int = 8
    grid_size: tuple[int, int] = (16, 16)
    checker_square_width: int = 2
    periodic: bool = False
    perception_kernels: int = 16
    perception_layers: tuple[int, ...] = (9, 8, 4, 2)
    update_layers: tuple[int, ...] = (
        264,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        128,
        64,
        32,
        16,
        8,
        8,
    )
    adam_b1: float = 0.9
    adam_b2: float = 0.99
    weight_decay: float = 0.01
    clip_value: float = 100.0


Params = dict[str, list[jax.Array]]
Wires = dict[str, list[list[jax.Array]]]


class TrainState(NamedTuple):
    params: Params
    opt_state: optax.OptState
    key: jax.Array
    update_index: jax.Array


def get_moore_connections(key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the source's shuffled neighbor-to-center Moore wiring."""
    neighbors = jnp.array([0, 1, 2, 3, 5, 6, 7, 8])
    a = neighbors
    b = jnp.full_like(neighbors, 4)
    perm = jax.random.permutation(key, neighbors.shape[0])
    return a[perm], b[perm]


def get_unique_connections(
    in_dim: int, out_dim: int, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Construct and shuffle the exact fixed-pair sequence used upstream."""
    if out_dim * 2 < in_dim:
        raise ValueError("number of outputs must be at least half the inputs")
    x = jnp.arange(in_dim)
    a = x[::2]
    b = x[1::2]
    size = min(a.shape[0], b.shape[0])
    a, b = a[:size], b[:size]
    if a.shape[0] < out_dim:
        a_extra = x[1::2]
        b_extra = x[2::2]
        size = min(a_extra.shape[0], b_extra.shape[0])
        a = jnp.concatenate([a, a_extra[:size]])
        b = jnp.concatenate([b, b_extra[:size]])
    offset = 2
    while out_dim > a.shape[0] and offset < in_dim:
        a = jnp.concatenate([a, x[:-offset]])
        b = jnp.concatenate([b, x[offset:]])
        offset += 1
    if a.shape[0] < out_dim:
        raise ValueError(f"could not generate {out_dim} connections")
    perm = jax.random.permutation(key, out_dim)
    return a[:out_dim][perm], b[:out_dim][perm]


def bin_op_all_combinations(a: jax.Array, b: jax.Array) -> jax.Array:
    """Evaluate gates 0..15, with columns ordered 00, 01, 10, 11."""
    return jnp.stack(
        [
            jnp.zeros_like(a),
            a * b,
            a - a * b,
            a,
            b - a * b,
            b,
            a + b - 2 * a * b,
            a + b - a * b,
            1 - (a + b - a * b),
            1 - (a + b - 2 * a * b),
            1 - b,
            1 - b + a * b,
            1 - a,
            1 - a + a * b,
            1 - a * b,
            jnp.ones_like(a),
        ],
        axis=-1,
    )


def bin_op_s(a: jax.Array, b: jax.Array, gate_weights: jax.Array) -> jax.Array:
    combinations = bin_op_all_combinations(a, b)
    return jnp.sum(combinations * gate_weights[None, ...], axis=-1)


def decode_soft(logits: jax.Array) -> jax.Array:
    return jax.nn.softmax(logits, axis=-1)


def decode_hard(logits: jax.Array) -> jax.Array:
    # Upstream relies on x64 being disabled, making one_hot FP32 implicitly.
    # State the dtype so FP64 diagnostic probes do not alter training dtypes.
    return jax.nn.one_hot(
        jnp.argmax(logits, axis=-1), NUMBER_OF_GATES, dtype=logits.dtype
    )


def init_gates(n: int) -> jax.Array:
    # The notebook runs with JAX's default x64-disabled policy. Keep training
    # parameters FP32 even when R000 enables x64 for small diagnostic probes.
    gates = jnp.zeros((n, NUMBER_OF_GATES), dtype=jnp.float32)
    return gates.at[:, PASS_THROUGH_GATE].set(DEFAULT_PASS_VALUE)


def init_gate_layer(
    key: jax.Array, in_dim: int, out_dim: int, connections: str
) -> tuple[jax.Array, list[jax.Array]]:
    if connections == "unique":
        indices_a, indices_b = get_unique_connections(in_dim, out_dim, key)
    elif connections == "first_kernel":
        indices_a, indices_b = get_moore_connections(key)
    else:
        raise ValueError(f"unsupported connection type: {connections}")
    return init_gates(out_dim), [indices_a, indices_b]


def _init_logic_gate_network(
    layers: tuple[int, ...], key: jax.Array
) -> tuple[list[jax.Array], list[list[jax.Array]]]:
    params: list[jax.Array] = []
    wires: list[list[jax.Array]] = []
    for in_dim, out_dim in pairwise(layers):
        key, subkey = jax.random.split(key)
        gate_logits, gate_wires = init_gate_layer(subkey, in_dim, out_dim, "unique")
        params.append(gate_logits)
        wires.append(gate_wires)
    return params, wires


def _init_perceive_network(
    layers: tuple[int, ...], n_kernels: int, key: jax.Array
) -> tuple[list[jax.Array], list[list[jax.Array]]]:
    params: list[jax.Array] = []
    wires: list[list[jax.Array]] = []
    connection_types = ("first_kernel", "unique", "unique")
    for (in_dim, out_dim), connection_type in zip(pairwise(layers), connection_types):
        key, subkey = jax.random.split(key)
        gate_logits, gate_wires = init_gate_layer(
            subkey, in_dim, out_dim, connection_type
        )
        params.append(
            gate_logits.repeat(n_kernels, axis=0).reshape(
                n_kernels, out_dim, NUMBER_OF_GATES
            )
        )
        wires.append(gate_wires)
    return params, wires


def init_diff_logic_ca(config: SyncConfig, key: jax.Array) -> tuple[Params, Wires]:
    key, update_key = jax.random.split(key)
    update_params, update_wires = _init_logic_gate_network(
        config.update_layers, update_key
    )
    key, perceive_key = jax.random.split(key)
    perceive_params, perceive_wires = _init_perceive_network(
        config.perception_layers, config.perception_kernels, perceive_key
    )
    return (
        {"update": update_params, "perceive": perceive_params},
        {"update": update_wires, "perceive": perceive_wires},
    )


def run_layer(
    logits: jax.Array,
    wires: list[jax.Array],
    x: jax.Array,
    training: bool | jax.Array,
) -> jax.Array:
    a = x[..., wires[0]]
    b = x[..., wires[1]]
    gate_weights = jax.lax.cond(training, decode_soft, decode_hard, logits)
    return bin_op_s(a, b, gate_weights)


def run_update(
    params: list[jax.Array],
    wires: list[list[jax.Array]],
    x: jax.Array,
    training: bool | jax.Array,
) -> jax.Array:
    for logits, layer_wires in zip(params, wires):
        x = run_layer(logits, layer_wires, x, training)
    return x


def run_perceive(
    params: list[jax.Array],
    wires: list[list[jax.Array]],
    x: jax.Array,
    training: bool | jax.Array,
) -> jax.Array:
    run_layer_map = jax.vmap(run_layer, in_axes=(0, None, 0, None))
    x_previous = x
    x = jnp.transpose(x, (1, 0))
    x = jnp.repeat(x[None, ...], params[0].shape[0], axis=0)
    for logits, layer_wires in zip(params, wires):
        x = run_layer_map(logits, layer_wires, x, training)
    # Exact einops source order: k c s -> (c s k).
    x = jnp.transpose(x, (1, 2, 0)).reshape(-1)
    return jnp.concatenate([x_previous[4, :], x], axis=-1)


def run_circuit(
    params: Params,
    wires: Wires,
    patch: jax.Array,
    training: bool | jax.Array,
) -> jax.Array:
    perceived = run_perceive(params["perceive"], wires["perceive"], patch, training)
    return run_update(params["update"], wires["update"], perceived, training)


@partial(jax.jit, static_argnums=(1, 2))
def get_grid_patches(
    grid: jax.Array, patch_size: int, channel_dim: int, periodic: bool | jax.Array
) -> jax.Array:
    pad_size = (patch_size - 1) // 2
    padded = jax.lax.cond(
        periodic,
        lambda value: jnp.pad(
            value,
            ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode="wrap",
        ),
        lambda value: jnp.pad(
            value,
            ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode="constant",
            constant_values=0,
        ),
        grid,
    )
    patches = jax.lax.conv_general_dilated_patches(
        padded[None, ...],
        filter_shape=(patch_size, patch_size),
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )[0]
    height, width = grid.shape[:2]
    return (
        patches.reshape(height, width, channel_dim, -1)
        .transpose(0, 1, 3, 2)
        .reshape(height * width, patch_size * patch_size, channel_dim)
    )


def run_sync(
    grid: jax.Array,
    params: Params,
    wires: Wires,
    training: bool | jax.Array,
    periodic: bool | jax.Array,
) -> jax.Array:
    patches = get_grid_patches(grid, 3, grid.shape[-1], periodic)
    updated = jax.vmap(run_circuit, in_axes=(None, None, 0, None))(
        params, wires, patches, training
    )
    return updated.reshape(grid.shape)


def rollout(
    grid: jax.Array,
    params: Params,
    wires: Wires,
    training: bool | jax.Array,
    periodic: bool | jax.Array,
    runtime_ticks: int,
) -> jax.Array:
    def body_fn(current: jax.Array, _unused: None) -> tuple[jax.Array, None]:
        return run_sync(current, params, wires, training, periodic), None

    final, _ = jax.lax.scan(body_fn, grid, None, length=runtime_ticks)
    return final


def rollout_trajectory(
    grid: jax.Array,
    params: Params,
    wires: Wires,
    training: bool | jax.Array,
    periodic: bool | jax.Array,
    runtime_ticks: int,
) -> jax.Array:
    def body_fn(current: jax.Array, _unused: None) -> tuple[jax.Array, jax.Array]:
        updated = run_sync(current, params, wires, training, periodic)
        return updated, updated

    _, states = jax.lax.scan(body_fn, grid, None, length=runtime_ticks)
    return jnp.concatenate([grid[None, ...], states], axis=0)


def batch_rollout(
    grids: jax.Array,
    params: Params,
    wires: Wires,
    training: bool | jax.Array,
    periodic: bool | jax.Array,
    runtime_ticks: int,
) -> jax.Array:
    return jax.vmap(rollout, in_axes=(0, None, None, None, None, None))(
        grids, params, wires, training, periodic, runtime_ticks
    )


def batch_trajectory(
    grids: jax.Array,
    params: Params,
    wires: Wires,
    training: bool | jax.Array,
    periodic: bool | jax.Array,
    runtime_ticks: int,
) -> jax.Array:
    trajectories = jax.vmap(
        rollout_trajectory, in_axes=(0, None, None, None, None, None)
    )(grids, params, wires, training, periodic, runtime_ticks)
    return jnp.swapaxes(trajectories, 0, 1)


def create_checkerboard(image_size: tuple[int, int], square_width: int) -> np.ndarray:
    height, width = image_size
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    return (((x // square_width) + (y // square_width)) % 2).astype(np.uint8)


def make_target(config: SyncConfig) -> jax.Array:
    image = create_checkerboard(config.grid_size, config.checker_square_width)
    target = jnp.zeros(
        (config.batch_size, *config.grid_size, config.channels), dtype=jnp.float32
    )
    return target.at[..., 0].set(image)


def sample_training_batch(
    data_key: jax.Array, config: SyncConfig
) -> tuple[jax.Array, jax.Array]:
    data_key, sample_key = jax.random.split(data_key)
    inputs = jax.random.randint(
        sample_key,
        (config.batch_size, *config.grid_size, config.channels),
        minval=0,
        maxval=2,
    ).astype(jnp.float32)
    return data_key, inputs


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


def init_train_state(
    config: SyncConfig, optimizer: optax.GradientTransformation
) -> tuple[TrainState, Wires]:
    key = jax.random.PRNGKey(config.seed)
    key, init_key = jax.random.split(key)
    params, wires = init_diff_logic_ca(config, init_key)
    return TrainState(params, optimizer.init(params), key, jnp.array(0)), wires


def loss(
    params: Params,
    wires: Wires,
    inputs: jax.Array,
    target: jax.Array,
    config: SyncConfig,
    training: bool,
) -> jax.Array:
    predicted = batch_rollout(
        inputs,
        params,
        wires,
        training,
        config.periodic,
        config.runtime_ticks,
    )
    return jnp.square(predicted[..., 0] - target[..., 0]).sum()


def make_train_step(config: SyncConfig, optimizer: optax.GradientTransformation):
    """Build the source-equivalent jitted step for one fixed configuration."""

    def loss_with_aux(
        params: Params,
        wires: Wires,
        inputs: jax.Array,
        target: jax.Array,
        _key: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        soft_loss = loss(params, wires, inputs, target, config, training=True)
        hard_loss = loss(params, wires, inputs, target, config, training=False)
        return soft_loss, {"hard": hard_loss}

    value_and_grad = jax.value_and_grad(loss_with_aux, argnums=0, has_aux=True)

    @jax.jit
    def train_step(
        state: TrainState,
        inputs: jax.Array,
        target: jax.Array,
        wires: Wires,
    ) -> tuple[TrainState, jax.Array, dict[str, jax.Array], Params]:
        key, loss_key = jax.random.split(state.key)
        (soft_loss, auxiliary), gradients = value_and_grad(
            state.params, wires, inputs, target, loss_key
        )
        updates, opt_state = optimizer.update(gradients, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        next_state = TrainState(params, opt_state, key, state.update_index + 1)
        return next_state, soft_loss, auxiliary, gradients

    return train_step


def replace_params(state: TrainState, params: Params) -> TrainState:
    return TrainState(params, state.opt_state, state.key, state.update_index)


def nontrivial_fixed_params(params: Params) -> Params:
    """Create deterministic, non-symmetric logits without consuming a PRNG key."""
    leaves, tree = jax.tree_util.tree_flatten(params)
    fixed: list[jax.Array] = []
    for leaf_index, leaf in enumerate(leaves):
        values = jnp.arange(leaf.size, dtype=jnp.float32).reshape(leaf.shape)
        values = jnp.sin(values * jnp.float32(0.017) + leaf_index * 0.31)
        fixed.append(values)
    return jax.tree_util.tree_unflatten(tree, fixed)


def flatten_float_tree(tree: Any) -> np.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate([np.asarray(leaf).reshape(-1) for leaf in leaves])


def hard_gate_ids(params: Params) -> Params:
    return jax.tree_util.tree_map(lambda value: jnp.argmax(value, axis=-1), params)


def expected_parameter_shapes(config: SyncConfig) -> dict[str, list[tuple[int, ...]]]:
    return {
        "perceive": [
            (config.perception_kernels, width, NUMBER_OF_GATES)
            for width in config.perception_layers[1:]
        ],
        "update": [(width, NUMBER_OF_GATES) for width in config.update_layers[1:]],
    }


def validate_config_contract(document: dict[str, Any]) -> SyncConfig:
    """Validate immutable scientific fields before constructing a run."""
    recipe = document["reference_recipe"]
    config = SyncConfig()
    expected = {
        "reference_seed": config.seed,
        "channels": config.channels,
        "batch_size": config.batch_size,
        "optimizer_updates": config.optimizer_updates,
        "runtime_ticks": config.runtime_ticks,
        "grid_size": list(config.grid_size),
        "checker_square_width": config.checker_square_width,
        "pass_through_gate_index": PASS_THROUGH_GATE,
        "initial_pass_through_logit": DEFAULT_PASS_VALUE,
        "periodic": config.periodic,
        "asynchronous": False,
    }
    mismatches = {
        name: (recipe.get(name), value)
        for name, value in expected.items()
        if recipe.get(name) != value
    }
    if recipe["perception"]["layer_widths_including_input"] != list(
        config.perception_layers
    ):
        mismatches["perception.layers"] = (
            recipe["perception"]["layer_widths_including_input"],
            list(config.perception_layers),
        )
    if recipe["update_layer_widths_including_input"] != list(config.update_layers):
        mismatches["update.layers"] = (
            recipe["update_layer_widths_including_input"],
            list(config.update_layers),
        )
    optimizer = recipe["optimizer"]
    optimizer_expected = {
        "learning_rate": config.learning_rate,
        "b1": config.adam_b1,
        "b2": config.adam_b2,
        "weight_decay": config.weight_decay,
    }
    for name, value in optimizer_expected.items():
        if optimizer.get(name) != value:
            mismatches[f"optimizer.{name}"] = (optimizer.get(name), value)
    if mismatches:
        raise ValueError(f"R001 config differs from the frozen recipe: {mismatches}")
    return config
