#!/usr/bin/env python3
"""Finite certificate for checkpoint 250 on its 16x16 zero-exterior grid.

0/1 are known bits; 2 represents the set {0,1}. Each abstract gate contains
every concrete possibility. A known output therefore holds for every initial
assignment, including correlated assignments. No new grids are sampled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_checkpoints import LAYERS, build_dag, decode, load_npz

NOT = np.array([1, 0, 2], dtype=np.uint8)
OR = np.array([[0, 1, 2], [1, 1, 1], [2, 1, 2]], dtype=np.uint8)
CONTENTS = [{0}, {1}, {0, 1}]


def step(v, m):
    """Synchronous core update, using exactly zero outside the 16x16 grid."""
    p, q = np.pad(m, 1), np.pad(v, 1)
    new_v = OR[p[2:, :-2], NOT[OR[p[:-2, :-2], p[2:, 2:]]]]
    new_m = q[2:, :-2]
    return new_v, new_m


def code(a):
    return "".join("01?"[int(x)] for x in a.ravel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expected = {
        "checkpoints/checkpoint_update_250.npz": "95eee045d2d78c2b533008e455f55d6199c66695e6679e098e39ea800da82f38",
        "analysis/final_hard_circuit.npz": "44cba42f059265d6bfeb21bfef0a9591b9739e28efa3633e7cf19db4f66eab71",
        "analysis/probe_trajectory_update_250.npz": "c8d3f4c98d6ba5c5609222b44967229797b0e63c20e5d0487743a0913daff156",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((args.artifact_dir / name).read_bytes()).hexdigest() == digest

    # Soundness of the abstract truth tables, checked on their finite domains.
    for a in range(3):
        assert {1 - x for x in CONTENTS[a]} == CONTENTS[NOT[a]]
        for b in range(3):
            assert {x | y for x in CONTENTS[a] for y in CONTENTS[b]} == CONTENTS[OR[a, b]]

    checkpoint = load_npz(args.artifact_dir / "checkpoints/checkpoint_update_250.npz")
    export = load_npz(args.artifact_dir / "analysis/final_hard_circuit.npz")
    wires = [(export[f"wire_{layer}_a"], export[f"wire_{layer}_b"]) for layer in LAYERS]
    dag = build_dag(decode(checkpoint)[3], wires)
    assert dag.summary()["visible_recurrent_channel_closure"] == [0, 2]
    variables = sorted({dag.nodes[i][1] for i in dag.reachable() if dag.nodes[i][0] == "input"})
    patch = np.zeros((2**len(variables), 72), dtype=np.uint8)
    patch[:, variables] = (np.arange(len(patch), dtype=np.uint32)[:, None] >> np.arange(len(variables))) & 1
    patch = patch.reshape(-1, 9, 8)
    output = dag.evaluate(patch)
    assert np.array_equal(output[:, 0], patch[:, 6, 2] | (1 - (patch[:, 0, 2] | patch[:, 8, 2])))
    assert np.array_equal(output[:, 2], patch[:, 6, 0])

    # Check spatial indexing/timing against the original recorded trajectory.
    saved = load_npz(args.artifact_dir / "analysis/probe_trajectory_update_250.npz")["hard"][:, 0].astype(np.uint8)
    v, m = saved[0, ..., 0], saved[0, ..., 2]
    for tick in range(1, 21):
        v, m = step(v, m)
        assert np.array_equal(v, saved[tick, ..., 0])
        assert np.array_equal(m, saved[tick, ..., 2])

    y, x = np.indices((16, 16))
    target = ((y // 2 + x // 2) % 2).astype(np.uint8)
    v = np.full((16, 16), 2, dtype=np.uint8)
    m = v.copy()
    trajectory = []
    for tick in range(18):
        known = v != 2
        assert not np.any(known & (v != target))
        trajectory.append({"tick": tick, "visible_known": int(known.sum()),
                           "relay_known": int((m != 2).sum()),
                           "visible": code(v), "relay": code(m)})
        if tick == 16:
            assert np.array_equal(v, target)
        if tick == 17:
            assert np.array_equal(v, target) and np.all(m != 2)
            next_v, next_m = step(v, m)
            assert np.array_equal(v, next_v) and np.array_equal(m, next_m)
        else:
            v, m = step(v, m)

    result = {
        "checkpoint": 250, "grid": [16, 16], "boundary": "constant zero",
        "update_schedule": "synchronous", "core_channels": [0, 2],
        "scope": "all 2^512 core initial states; other six channels cannot affect the core",
        "method": "sound set-valued Boolean propagation, starting every interior bit unknown",
        "visible_target_guaranteed_at_tick": 16,
        "core_fixed_point_guaranteed_at_tick": 17,
        "visible_target_remains_forever": True,
        "local_rule_matches_exported_wiring": True,
        "local_assignments_checked": len(patch),
        "selected_saved_trajectory_matches": True,
        "input_sha256": expected,
        "abstract_symbols": {"0": "known zero", "1": "known one", "?": "either bit"},
        "abstract_trajectory": trajectory,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "abstract_trajectory"}, indent=2))


if __name__ == "__main__":
    main()
