from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recurrent_circuit_learning.notebook_oracle import (
    NOTEBOOK_CELL_INDICES,
    load_notebook_namespace,
)

ROOT = Path(__file__).parents[1]


def test_oracle_executes_only_recorded_vendored_definition_cells() -> None:
    namespace, record = load_notebook_namespace(
        ROOT / "references/difflogic_ca/upstream/diffLogic_CA.ipynb"
    )
    assert (
        tuple(cell["index"] for cell in record["executed_cells"])
        == NOTEBOOK_CELL_INDICES
    )
    assert [cell["id"] for cell in record["executed_cells"]] == [
        "FYKUs0uhu78Q",
        "bNkdywCOEir4",
        "0coNJrpHEkPx",
        "rblos5LlWJWn",
        "_3AdVC_TWQ1V",
    ]
    assert "recurrent_circuit_learning" not in "\n".join(
        cell["source"] for cell in record["executed_cells"]
    )
    pairs = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
    actual = namespace["bin_op_all_combinations"](pairs[:, 0], pairs[:, 1]).T
    expected = np.array(
        [[(gate >> (3 - column)) & 1 for column in range(4)] for gate in range(16)],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)
