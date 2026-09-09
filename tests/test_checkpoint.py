from pathlib import Path

import jax
import numpy as np

from recurrent_circuit_learning.checkpoint import load_checkpoint, save_checkpoint
from recurrent_circuit_learning.difflogic_ca import (
    SyncConfig,
    flatten_float_tree,
    init_train_state,
    make_optimizer,
)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = SyncConfig()
    optimizer = make_optimizer(config)
    state, _ = init_train_state(config, optimizer)
    data_key = jax.random.PRNGKey(123)
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, state, data_key, {"test": True})
    restored, restored_data_key, metadata = load_checkpoint(path, config, optimizer)
    np.testing.assert_array_equal(
        flatten_float_tree(restored.params), flatten_float_tree(state.params)
    )
    np.testing.assert_array_equal(restored.key, state.key)
    np.testing.assert_array_equal(restored_data_key, data_key)
    assert int(restored.update_index) == 0
    assert metadata == {"test": True}
