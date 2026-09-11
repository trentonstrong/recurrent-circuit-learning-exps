from __future__ import annotations

import json
from pathlib import Path

import jax
import pytest

if not jax.config.x64_enabled:
    pytest.skip("R003 numerical controls require JAX_ENABLE_X64=1", allow_module_level=True)

import jax.numpy as jnp
import numpy as np

from recurrent_circuit_learning.r003 import (
    T64,
    effective_q_from_logits,
    eligibility_mask,
    factorized_probabilities,
    local_geometry,
    replace_logits,
    validate_config_contract,
)

ROOT = Path(__file__).parents[1]


def test_r003_config_is_frozen_v1_protocol() -> None:
    document = json.loads(
        (ROOT / "configs/experiments/r003_same_function_v1.json").read_text()
    )
    validate_config_contract(document)


def test_truth_table_mapping_has_rank_five_with_normalization() -> None:
    matrix = np.concatenate([np.ones((1, 16)), np.asarray(T64).T], axis=0)
    assert np.linalg.matrix_rank(matrix) == 5


def test_factorized_replacement_preserves_q_and_logit_gauge() -> None:
    logits = jnp.sin(jnp.arange(48, dtype=jnp.float64).reshape(3, 16) * 0.17)
    source_q = effective_q_from_logits(logits)
    source_mean = logits.mean(axis=-1)
    for alpha in (0.0, 0.5, 1.0):
        replaced, eligible = replace_logits(logits, alpha, 1e-8)
        assert np.all(np.asarray(eligible))
        np.testing.assert_allclose(
            effective_q_from_logits(replaced), source_q, atol=1e-12, rtol=1e-10
        )
        np.testing.assert_allclose(replaced.mean(axis=-1), source_mean, atol=1e-12)


def test_factorizing_a_factorized_distribution_is_identity() -> None:
    q = jnp.array([[0.1, 0.3, 0.7, 0.9]], dtype=jnp.float64)
    p = factorized_probabilities(q)
    np.testing.assert_allclose(p @ T64, q, atol=1e-14)
    np.testing.assert_allclose(factorized_probabilities(p @ T64), p, atol=1e-14)


def test_and_or_and_copy_mixtures_have_same_q_but_different_geometry() -> None:
    uniform = np.full(16, 1 / 16)
    first = np.zeros(16); first[[1, 7]] = 0.5
    second = np.zeros(16); second[[3, 5]] = 0.5
    z1 = jnp.log(jnp.asarray(0.8 * first + 0.2 * uniform))
    z2 = jnp.log(jnp.asarray(0.8 * second + 0.2 * uniform))
    h = jnp.array([0.2, -0.3, 0.4, -0.1], dtype=jnp.float64)
    q1, q2 = effective_q_from_logits(z1), effective_q_from_logits(z2)
    np.testing.assert_allclose(q1, [0.1, 0.5, 0.5, 0.9], atol=1e-15)
    np.testing.assert_allclose(q1, q2, atol=1e-15)
    _, j1, m1, _, v1 = local_geometry(z1, h)
    _, j2, m2, _, v2 = local_geometry(z2, h)
    np.testing.assert_allclose(j1, jax.jacrev(effective_q_from_logits)(z1), atol=1e-12)
    np.testing.assert_allclose(j2, jax.jacrev(effective_q_from_logits)(z2), atol=1e-12)
    assert not np.allclose(m1, m2)
    assert not np.allclose(v1, v2)


def test_eligibility_rejects_boundary_q() -> None:
    interior = jnp.zeros((1, 16), dtype=jnp.float64)
    boundary = jnp.full((1, 16), -1000.0, dtype=jnp.float64).at[:, 0].set(0)
    assert bool(eligibility_mask(interior, 1e-8)[0])
    assert not bool(eligibility_mask(boundary, 1e-8)[0])
