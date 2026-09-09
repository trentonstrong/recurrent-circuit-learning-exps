from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from recurrent_circuit_learning.difflogic_ca import (
    PASS_THROUGH_GATE,
    SyncConfig,
    bin_op_all_combinations,
    expected_parameter_shapes,
    get_grid_patches,
    init_train_state,
    make_optimizer,
    validate_config_contract,
)

ROOT = Path(__file__).parents[1]


def test_all_boolean_gate_tables_and_a_orientation() -> None:
    pairs = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
    actual = np.asarray(bin_op_all_combinations(pairs[:, 0], pairs[:, 1])).T
    expected = np.array(
        [[(gate >> (3 - column)) & 1 for column in range(4)] for gate in range(16)],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[PASS_THROUGH_GATE], [0, 0, 1, 1])


def test_fractional_extensions_are_multilinear() -> None:
    a = jnp.array([0.25], dtype=jnp.float64)
    b = jnp.array([0.75], dtype=jnp.float64)
    values = np.asarray(bin_op_all_combinations(a, b))[0]
    assert values[1] == 0.1875
    assert values[3] == 0.25
    assert values[5] == 0.75
    assert values[6] == 0.625


def test_zero_boundary_patch_order_is_filter_then_channel() -> None:
    grid = jnp.arange(1, 13, dtype=jnp.float32).reshape(2, 3, 2)
    actual = np.asarray(get_grid_patches(grid, 3, 2, False))
    expected_first = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 2],
            [3, 4],
            [0, 0],
            [7, 8],
            [9, 10],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual[0], expected_first)
    assert actual.shape == (6, 9, 2)


def test_reference_shapes_wiring_and_parameter_counts() -> None:
    config = SyncConfig()
    state, wires = init_train_state(config, make_optimizer(config))
    shapes = {
        name: [tuple(value.shape) for value in state.params[name]]
        for name in ("perceive", "update")
    }
    assert shapes == expected_parameter_shapes(config)
    assert (
        sum(value.shape[0] * value.shape[1] for value in state.params["perceive"])
        == 224
    )
    assert sum(value.shape[-2] for value in state.params["update"]) == 2816
    assert len(wires["perceive"]) == 3
    assert len(wires["update"]) == 16
    assert all(
        value.dtype == jnp.float32 for value in jax.tree_util.tree_leaves(state.params)
    )


def test_machine_readable_config_matches_implementation() -> None:
    document = json.loads(
        (ROOT / "configs/experiments/r001_sync_reference.json").read_text()
    )
    assert validate_config_contract(document) == SyncConfig()
