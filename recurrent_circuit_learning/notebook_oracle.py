"""Execute the vendored DiffLogic CA notebook definitions as an R000 oracle.

Only definition cells needed by the synchronous checkerboard recipe are executed.
The extracted implementation is deliberately not imported here.
"""

from __future__ import annotations

import hashlib
import json
from collections import namedtuple
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.lax import conv_general_dilated_patches

NOTEBOOK_CELL_INDICES = (6, 8, 10, 12, 16)


def _rearrange(value: jax.Array, pattern: str, **axes: int) -> jax.Array:
    """Minimal adapters for the two einops expressions in executed cells."""
    if pattern == "k c s -> (c s k)":
        return jnp.transpose(value, (1, 2, 0)).reshape(-1)
    if pattern == "x y (c f) -> (x y) f c":
        channels = axes["c"]
        height, width, combined = value.shape
        filters = combined // channels
        return (
            value.reshape(height, width, channels, filters)
            .transpose(0, 1, 3, 2)
            .reshape(height * width, filters, channels)
        )
    raise ValueError(f"unsupported notebook rearrange expression: {pattern}")


def synchronous_hyperparameters(seed: int = 23) -> dict[str, Any]:
    perceive = {
        "n_kernels": 16,
        "layers": [9, 8, 4, 2],
        "connections": ["first_kernel", "unique", "unique", "unique"],
    }
    channels = 8
    update_input = perceive["n_kernels"] * channels * perceive["layers"][-1] + channels
    return {
        "seed": seed,
        "lr": 0.05,
        "batch_size": 2,
        "num_epochs": 500,
        "num_steps": 20,
        "channels": channels,
        "periodic": 0,
        "perceive": perceive,
        "update": {
            "layers": [update_input] + [256] * 10 + [128, 64, 32, 16, 8, channels],
            "connections": ["unique"] * 17,
        },
        "async_training": False,
    }


def load_notebook_namespace(
    notebook_path: Path, seed: int = 23
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the exact selected cell sources in a fresh supplied namespace."""
    document = json.loads(notebook_path.read_text())
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "jax": jax,
        "jnp": jnp,
        "np": np,
        "optax": optax,
        "random": jax.random,
        "partial": partial,
        "namedtuple": namedtuple,
        "conv_general_dilated_patches": conv_general_dilated_patches,
        "rearrange": _rearrange,
        "nn": SimpleNamespace(softmax=jax.nn.softmax),
        "hyperparams": synchronous_hyperparameters(seed),
    }
    executed: list[dict[str, Any]] = []
    for index in NOTEBOOK_CELL_INDICES:
        cell = document["cells"][index]
        source = "".join(cell["source"])
        cell_id = cell.get("id") or cell.get("metadata", {}).get("id")
        exec(  # noqa: S102 - executing the audited vendored source is the oracle.
            compile(source, f"{notebook_path.name}:cell-{index}:{cell_id}", "exec"),
            namespace,
        )
        executed.append(
            {
                "index": index,
                "id": cell_id,
                "source": source,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        )
    record = {
        "notebook_path": str(notebook_path),
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "executed_cells": executed,
        "supplied_globals": sorted(
            name for name in namespace if not name.startswith("__")
        ),
        "adapters": {
            "flax.linen.softmax": "jax.nn.softmax",
            "einops.rearrange": [
                "k c s -> (c s k): transpose(1,2,0), reshape(-1)",
                "x y (c f) -> (x y) f c: explicit reshape/transpose",
            ],
            "checkerboard_setup": "frozen synchronous hyperparameters supplied before cell 16",
        },
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
    }
    return namespace, record
