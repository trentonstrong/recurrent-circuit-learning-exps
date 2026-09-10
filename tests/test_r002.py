from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from recurrent_circuit_learning.difflogic_ca import bin_op_s, decode_soft
from recurrent_circuit_learning.r002 import (
    CONDITIONS,
    GATE_TRUTH_TABLES,
    categorical_to_truth_params,
    config_for,
    effective_truth_tables,
    gate_ids,
    init_condition_state,
    init_paired,
    load_checkpoint,
    make_optimizer,
    multilinear_gate,
    save_checkpoint,
    tree_sha256,
    validate_config_contract,
)

ROOT = Path(__file__).parents[1]


def test_r002_config_is_the_frozen_four_condition_cohort() -> None:
    document = json.loads(
        (ROOT / "configs/experiments/r002_paired_gate_coordinates.json").read_text()
    )
    seeds, conditions = validate_config_contract(document)
    assert seeds == tuple(range(16))
    assert conditions == CONDITIONS


def test_categorical_and_truth_coordinates_start_at_same_function() -> None:
    paired = init_paired(23)
    categorical_q = effective_truth_tables(paired.categorical_params, "categorical")
    truth_q = effective_truth_tables(paired.truth_params, "truth")
    for categorical, truth in zip(
        jax.tree_util.tree_leaves(categorical_q),
        jax.tree_util.tree_leaves(truth_q),
    ):
        np.testing.assert_allclose(categorical, truth, atol=1e-7, rtol=1e-6)


def test_common_multilinear_kernel_matches_source_gate_mixture() -> None:
    logits = jnp.sin(jnp.arange(80, dtype=jnp.float32).reshape(5, 16) * 0.17)
    a = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32)
    b = jnp.linspace(1.0, 0.0, 5, dtype=jnp.float32)
    q = jnp.matmul(
        decode_soft(logits),
        GATE_TRUTH_TABLES,
        precision=jax.lax.Precision.HIGHEST,
    )
    expected = bin_op_s(a, b, decode_soft(logits))
    np.testing.assert_allclose(
        multilinear_gate(a, b, q), expected, atol=2e-7, rtol=2e-6
    )


def test_truth_hardening_is_four_thresholds_not_argmax() -> None:
    q = (GATE_TRUTH_TABLES * 0.8 + 0.1).astype(jnp.float32)
    logits = jnp.log(q) - jnp.log1p(-q)
    params = {"perceive": [logits], "update": [logits]}
    ids = gate_ids(params, "truth", "native")
    np.testing.assert_array_equal(ids["perceive"][0], np.arange(16))
    np.testing.assert_array_equal(ids["update"][0], np.arange(16))


def test_checkpoint_rejects_the_wrong_representation(tmp_path: Path) -> None:
    condition = CONDITIONS["truth_no_decay"]
    config = config_for(23, condition, optimizer_updates=2)
    optimizer = make_optimizer(config)
    state, _, _ = init_condition_state(23, condition, optimizer)
    metadata = {"condition": condition.name, "representation": "truth"}
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, state, jax.random.PRNGKey(23), metadata)
    restored, restored_key, restored_metadata = load_checkpoint(path, state, metadata)
    assert tree_sha256(restored.params) == tree_sha256(state.params)
    np.testing.assert_array_equal(restored_key, jax.random.PRNGKey(23))
    assert restored_metadata == metadata
    with pytest.raises(ValueError, match="representation"):
        load_checkpoint(path, state, {"representation": "categorical"})


def test_truth_conversion_has_no_hidden_clamp() -> None:
    logits = {"perceive": [jnp.zeros((1, 16), dtype=jnp.float32)], "update": []}
    truth = categorical_to_truth_params(logits)
    expected_q = effective_truth_tables(logits, "categorical")["perceive"][0]
    np.testing.assert_array_equal(jax.nn.sigmoid(truth["perceive"][0]), expected_q)
