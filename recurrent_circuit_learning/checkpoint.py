"""Small, explicit NumPy checkpoints for exact experiment resumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .difflogic_ca import SyncConfig, TrainState, init_train_state


def save_checkpoint(
    path: Path,
    state: TrainState,
    data_key: jax.Array,
    metadata: dict[str, Any] | None = None,
) -> None:
    param_leaves, _ = jax.tree_util.tree_flatten(state.params)
    opt_leaves, _ = jax.tree_util.tree_flatten(state.opt_state)
    arrays: dict[str, Any] = {
        "format_version": np.array(1, dtype=np.int64),
        "update_index": np.asarray(state.update_index),
        "model_key": np.asarray(state.key),
        "data_key": np.asarray(data_key),
        "param_leaf_count": np.array(len(param_leaves), dtype=np.int64),
        "opt_leaf_count": np.array(len(opt_leaves), dtype=np.int64),
        "metadata_json": np.array(json.dumps(metadata or {}, sort_keys=True)),
    }
    arrays.update(
        {
            f"param_{index:03d}": np.asarray(value)
            for index, value in enumerate(param_leaves)
        }
    )
    arrays.update(
        {
            f"opt_{index:03d}": np.asarray(value)
            for index, value in enumerate(opt_leaves)
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_checkpoint(
    path: Path,
    config: SyncConfig,
    optimizer: optax.GradientTransformation,
) -> tuple[TrainState, jax.Array, dict[str, Any]]:
    template, _ = init_train_state(config, optimizer)
    _, param_tree = jax.tree_util.tree_flatten(template.params)
    _, opt_tree = jax.tree_util.tree_flatten(template.opt_state)
    with np.load(path, allow_pickle=False) as stored:
        if int(stored["format_version"]) != 1:
            raise ValueError("unsupported checkpoint format")
        param_count = int(stored["param_leaf_count"])
        opt_count = int(stored["opt_leaf_count"])
        params = jax.tree_util.tree_unflatten(
            param_tree,
            [jnp.asarray(stored[f"param_{i:03d}"]) for i in range(param_count)],
        )
        opt_state = jax.tree_util.tree_unflatten(
            opt_tree, [jnp.asarray(stored[f"opt_{i:03d}"]) for i in range(opt_count)]
        )
        state = TrainState(
            params,
            opt_state,
            jnp.asarray(stored["model_key"]),
            jnp.asarray(stored["update_index"]),
        )
        data_key = jnp.asarray(stored["data_key"])
        metadata = json.loads(str(stored["metadata_json"]))
    return state, data_key, metadata
