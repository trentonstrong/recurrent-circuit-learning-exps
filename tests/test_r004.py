from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from recurrent_circuit_learning.difflogic_ca import make_target, sample_training_batch
from recurrent_circuit_learning.r002 import (
    CONDITIONS,
    config_for,
    init_condition_state,
    load_checkpoint,
    make_optimizer,
    make_train_step,
)
from recurrent_circuit_learning.r004_analysis import (
    _binomial_two_sided,
    _holm_two,
    _write_jsonl,
)
from recurrent_circuit_learning.r004_runner import (
    _archive_superseded_tail,
    batch_sha256,
    checkpoint_metadata,
    compare_states,
    continuation_batch_hashes_sha256,
    load_parent,
    save_checkpoint_atomic,
    validate_config_contract,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs/experiments/r004_training_continuation_v1.json"


def test_r004_config_is_the_frozen_continuation_cohort() -> None:
    document = json.loads(CONFIG_PATH.read_text())
    seeds, conditions = validate_config_contract(document)
    assert seeds == tuple(range(16))
    assert conditions == CONDITIONS

    changed = json.loads(CONFIG_PATH.read_text())
    changed["scientific_recipe"]["runtime_ticks"] = 21
    with pytest.raises(ValueError, match="runtime_ticks"):
        validate_config_contract(changed)


def test_indexed_batch_digest_has_global_update_identity() -> None:
    batch = np.arange(12, dtype=np.uint8).reshape(3, 4)
    expected = hashlib.sha256()
    expected.update(b"uint8")
    expected.update(b"[3, 4]")
    expected.update(batch.tobytes(order="C"))
    assert batch_sha256(batch) == expected.hexdigest()

    records = [
        {"global_update": 501, "training_batch_sha256": "a" * 64},
        {"global_update": 502, "training_batch_sha256": "b" * 64},
    ]
    expected_stream = hashlib.sha256(
        f"501:{'a' * 64}\n502:{'b' * 64}\n".encode("ascii")
    ).hexdigest()
    assert continuation_batch_hashes_sha256(records) == expected_stream
    records[1]["global_update"] = 503
    with pytest.raises(ValueError, match="non-contiguous"):
        continuation_batch_hashes_sha256(records)


def test_budget_field_does_not_change_the_r002_step() -> None:
    condition = CONDITIONS["truth_no_decay"]
    old_config = config_for(23, condition, optimizer_updates=500)
    new_config = config_for(23, condition, optimizer_updates=2000)
    old_optimizer = make_optimizer(old_config)
    new_optimizer = make_optimizer(new_config)
    old_state, wires, _ = init_condition_state(23, condition, old_optimizer)
    new_state, _, _ = init_condition_state(23, condition, new_optimizer)
    data_key = jax.random.PRNGKey(23)
    _, inputs = sample_training_batch(data_key, old_config)
    old_result = make_train_step(old_config, condition.representation, old_optimizer)(
        old_state, inputs, make_target(old_config), wires
    )
    new_result = make_train_step(new_config, condition.representation, new_optimizer)(
        new_state, inputs, make_target(new_config), wires
    )
    assert all(compare_states(old_result[0], new_result[0]).values())
    np.testing.assert_array_equal(old_result[1], new_result[1])


def test_r004_checkpoint_is_atomic_and_rejects_parent_mismatch(tmp_path: Path) -> None:
    condition = CONDITIONS["categorical_no_decay"]
    parent = load_parent(ROOT, ROOT / "artifacts", 0, condition)
    metadata = checkpoint_metadata(
        config_sha256=hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        seed=0,
        condition=condition,
        parent=parent,
    )
    path = tmp_path / "checkpoint_update_0500.npz"
    save_checkpoint_atomic(path, parent.state, parent.data_key, metadata)
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp.npz"))
    restored, restored_key, restored_metadata = load_checkpoint(
        path, parent.state, metadata
    )
    assert all(compare_states(parent.state, restored).values())
    np.testing.assert_array_equal(restored_key, parent.data_key)
    assert restored_metadata == metadata
    with pytest.raises(ValueError, match="parent_checkpoint_sha256"):
        load_checkpoint(
            path,
            parent.state,
            {"parent_checkpoint_sha256": "wrong"},
        )


def test_parent_loader_uses_saved_state_and_verified_wiring() -> None:
    condition = CONDITIONS["categorical_reference_decay"]
    parent = load_parent(ROOT, ROOT / "artifacts", 0, condition)
    assert int(parent.state.update_index) == 500
    assert parent.manifest["pairing"]["wiring_sha256"]
    assert all(
        np.asarray(leaf).dtype == np.float32
        for leaf in jax.tree_util.tree_leaves(parent.state.params)
    )


def test_superseded_tail_is_archived_outside_authoritative_metrics(
    tmp_path: Path,
) -> None:
    records = [
        {
            "global_update": update,
            "training_batch_sha256": f"{update:064x}"[-64:],
        }
        for update in range(501, 504)
    ]
    retained = _archive_superseded_tail(tmp_path, records, 502)
    assert [record["global_update"] for record in retained] == [501, 502]
    archived = [
        json.loads(line)
        for line in (tmp_path / "attempt_history.jsonl").read_text().splitlines()
    ]
    assert archived[0]["status"] == "superseded_uncheckpointed_tail"
    assert archived[0]["metric"]["global_update"] == 503


def test_exact_mcnemar_and_holm_helpers() -> None:
    assert _binomial_two_sided(0, 0) == 1.0
    assert _binomial_two_sided(0, 4) == 0.125
    adjusted = _holm_two({"reference_decay": 0.02, "no_decay": 0.5})
    assert adjusted == {"reference_decay": 0.04, "no_decay": 0.5}


def test_runtime_jsonl_compression_is_deterministic(tmp_path: Path) -> None:
    rows = [{"seed": 0, "value": [1, 2, 3]}]
    path = tmp_path / "runtime_profiles.jsonl.gz"
    _write_jsonl(path, rows)
    first = path.read_bytes()
    _write_jsonl(path, rows)
    assert path.read_bytes() == first
