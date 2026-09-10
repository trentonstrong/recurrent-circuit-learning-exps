from __future__ import annotations

import numpy as np

from scripts.r002_runtime_profile import (
    load_profile_cache,
    save_profile_cache,
    summarize_orbit,
    validate_cycle_bookkeeping,
)


def test_cycle_bookkeeping_regressions() -> None:
    checks = validate_cycle_bookkeeping()
    assert all(checks.values())


def test_cycle_phase_and_finite_suffix_are_distinct() -> None:
    mixed = summarize_orbit(
        [b"a", b"b", b"a"],
        [b"x", b"y", b"x"],
        [0, 1, 0],
    )
    assert mixed["cycle_behavior"] == "some_phases_correct"
    assert mixed["correct_phase_indices"] == [0]
    assert mixed["cycle_target_fraction"] == 0.5

    capped = summarize_orbit(
        [b"a", b"b", b"c"],
        [b"x", b"y", b"z"],
        [1, 1, 0],
    )
    assert capped["cycle_behavior"] == "unresolved_by_cap"
    assert capped["observed_correct_suffix_start"] == 2
    assert capped["observed_correct_suffix_length"] == 1


def test_profile_resume_cache_requires_matching_identity(tmp_path) -> None:
    arrays = {
        "errors": np.zeros((66, 257), dtype=np.int16),
        "visible_hamming_changes": np.zeros((66, 257), dtype=np.int16),
        "full_state_hamming_changes": np.zeros((66, 257), dtype=np.int16),
    }
    save_profile_cache(
        tmp_path,
        "rule",
        "config",
        "inputs",
        "implementation",
        {"run_id": "source"},
        arrays,
    )

    loaded = load_profile_cache(
        tmp_path, "rule", "config", "inputs", "implementation"
    )
    assert loaded is not None
    assert loaded[0]["run_id"] == "source"
    assert np.array_equal(loaded[1]["errors"], arrays["errors"])
    assert (
        load_profile_cache(
            tmp_path, "rule", "different", "inputs", "implementation"
        )
        is None
    )
    assert load_profile_cache(tmp_path, "rule", "config", "inputs", "changed") is None
