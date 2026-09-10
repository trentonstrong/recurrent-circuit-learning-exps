#!/usr/bin/env python3
"""Post-hoc R001 checkpoint review, independent NumPy Boolean execution.

No optimization or sampling is performed. Boolean simplification is exact but
is not minimization: propagate constants/copies/complement edges and merge
identical AND/XOR expressions. Training parameter entropy is not code length.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

RUN = "R001_20260909T231016Z_seed23_jaxgpu_attempt01"
BUNDLE_HASHES = {
    "checkpoints": "cff77e87faf6653ff7561c46a219ea2a3d4c1ee3af8080ba9b5c91324240bd95",
    "analysis": "d30d67c104729c72465eec5fa223cdf5c9eef479a440cbeca407162a7c7b739b",
}
T = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8)
LAYERS = [f"perceive_{i:02d}" for i in range(3)] + [f"update_{i:02d}" for i in range(16)]
GATE_NAMES = ["0", "AND", "A & ~B", "A", "~A & B", "B", "XOR", "OR",
              "NOR", "XNOR", "~B", "A | ~B", "~A", "~A | B", "NAND", "1"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_bundles(bundle_dir, artifact_dir, manifest):
    """Read verified regular files by name; never extract arbitrary archive paths."""
    manifest_records = {Path(a["path"]).name: a for a in manifest["artifacts"]}
    for checkpoint in manifest["checkpoints"]:
        for key in ("checkpoint", "probe_trajectory"):
            a = checkpoint[key]
            manifest_records[Path(a["path"]).name] = a
    verified = []
    for kind, expected in BUNDLE_HASHES.items():
        source = bundle_dir / f"{RUN}_{kind}.tar.gz"
        data = source.read_bytes()
        assert sha256(data) == expected, source
        dst = artifact_dir / kind
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(source, "r:gz") as archive:
            contents = json.load(archive.extractfile("CONTENTS.json"))
            assert contents["run_id"] == RUN and contents["experiment_id"] == "R001"
            assert contents["bundle_kind"] == kind
            files = contents["files"]
            assert len(archive.getmembers()) == len(files) + 1
            assert {m.name for m in archive.getmembers()} == {"CONTENTS.json"} | {f["name"] for f in files}
            for record in files:
                name = record["name"]
                assert Path(name).name == name
                member = archive.getmember(name)
                assert member.isfile()
                value = archive.extractfile(member).read()
                original = manifest_records[name]
                assert len(value) == record["bytes"] == original["bytes"]
                assert sha256(value) == record["sha256"] == original["sha256"]
                (dst / name).write_bytes(value)
            verified.append({"name": source.name, "bytes": len(data), "sha256": expected,
                             "verified_constituent_files": len(files)})
    return verified


def load_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return {k: file[k] for k in file.files}


def decode(checkpoint):
    params = [checkpoint[f"param_{i:03d}"].astype(np.float64) for i in range(19)]
    probabilities, truths, native, rounded = [], [], [], []
    for z in params:
        p = np.exp(z - z.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        q = p @ T
        probabilities.append(p)
        truths.append(q)
        native.append(z.argmax(-1).astype(np.uint8))
        rounded.append(((q >= .5) * [8, 4, 2, 1]).sum(-1).astype(np.uint8))
    return params, probabilities, truths, native, rounded


def patches(grid):
    batch, height, width, channels = grid.shape
    padded = np.pad(grid, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant")
    return np.stack([padded[:, y:y+height, x:x+width, :] for y in range(3) for x in range(3)], axis=-2)


def boolean_gate(g, a, b):
    return (g >> (3 - (2 * a + b))) & 1


def full_step(grid, gates, wires):
    """Full unsimplified 4,608-instance local network, uint8 truth lookup."""
    patch = patches(grid)
    x = np.swapaxes(patch, -1, -2)[..., None, :, :]  # B,H,W,K=1,C,9
    for i in range(3):
        a, b = wires[i]
        x = boolean_gate(gates[i][:, None, :], x[..., a], x[..., b])
    # B,H,W,K,C,S -> B,H,W,C,S,K, as in source einops k c s -> (c s k).
    perceived = x.transpose(0, 1, 2, 4, 5, 3).reshape(*grid.shape[:3], -1)
    x = np.concatenate([grid, perceived], axis=-1)
    for i in range(3, 19):
        a, b = wires[i]
        x = boolean_gate(gates[i], x[..., a], x[..., b])
    return x


class BooleanDAG:
    """AND/XOR DAG with complemented edges; literal 0/1 are constants."""
    def __init__(self):
        self.nodes = [("constant",)] + [("input", j) for j in range(72)]
        self.cache = {}
        self.outputs = []

    def conjunction(self, a, b):
        if a == 0 or b == 0 or a == (b ^ 1):
            return 0
        if a == 1:
            return b
        if b == 1 or a == b:
            return a
        return self.node("AND", a, b)

    def xor(self, a, b):
        sign = (a & 1) ^ (b & 1)
        a &= ~1
        b &= ~1
        if a == b:
            return sign
        if a == 0:
            return b ^ sign
        if b == 0:
            return a ^ sign
        return self.node("XOR", a, b) ^ sign

    def node(self, op, a, b):
        key = (op, min(a, b), max(a, b))
        if key not in self.cache:
            self.cache[key] = 2 * len(self.nodes)
            self.nodes.append(key)
        return self.cache[key]

    def gate(self, gate_id, a, b):
        # All 16 cases are checked against the truth table in validate_algebra().
        if gate_id == 0: return 0
        if gate_id == 1: return self.conjunction(a, b)
        if gate_id == 2: return self.conjunction(a, b ^ 1)
        if gate_id == 3: return a
        if gate_id == 4: return self.conjunction(a ^ 1, b)
        if gate_id == 5: return b
        if gate_id == 6: return self.xor(a, b)
        if gate_id == 7: return self.conjunction(a ^ 1, b ^ 1) ^ 1
        if gate_id == 8: return self.conjunction(a ^ 1, b ^ 1)
        if gate_id == 9: return self.xor(a, b) ^ 1
        if gate_id == 10: return b ^ 1
        if gate_id == 11: return self.conjunction(a ^ 1, b) ^ 1
        if gate_id == 12: return a ^ 1
        if gate_id == 13: return self.conjunction(a, b ^ 1) ^ 1
        if gate_id == 14: return self.conjunction(a, b) ^ 1
        if gate_id == 15: return 1
        raise ValueError(gate_id)

    def reachable(self, outputs=None):
        found = set()
        pending = list(self.outputs if outputs is None else outputs)
        while pending:
            index = pending.pop() // 2
            if index in found:
                continue
            found.add(index)
            if self.nodes[index][0] in ("AND", "XOR"):
                pending.extend(self.nodes[index][1:])
        return sorted(found)

    def evaluate(self, patch):
        values = {0: np.uint8(0)}
        x = patch.reshape(*patch.shape[:-2], 72)
        def val(literal):
            return values[literal // 2] ^ (literal & 1)
        for index in self.reachable():
            op, *args = self.nodes[index]
            if op == "input":
                values[index] = x[..., args[0]]
            elif op == "AND":
                values[index] = val(args[0]) & val(args[1])
            elif op == "XOR":
                values[index] = val(args[0]) ^ val(args[1])
        return np.stack([np.broadcast_to(val(o), x.shape[:-1]) for o in self.outputs], axis=-1)

    def expression(self, literal):
        index, inv = divmod(int(literal), 2)
        op, *args = self.nodes[index]
        if op == "constant":
            return str(inv)
        if op == "input":
            offset, channel = divmod(args[0], 8)
            dy, dx = divmod(offset, 3)
            expr = f"s{channel}[{dy-1:+d},{dx-1:+d}]"
        else:
            # Canonicalize commutative expressions across separately built DAGs.
            a, b = sorted([self.expression(args[0]), self.expression(args[1])])
            expr = f"({a} {'&' if op == 'AND' else '^'} {b})"
        return f"~{expr}" if inv else expr

    def summary(self):
        reachable = self.reachable()
        used_inputs = [self.nodes[i][1] for i in reachable if self.nodes[i][0] == "input"]
        closure = {0}
        while True:
            cone = self.reachable([self.outputs[c] for c in closure])
            expanded = closure | {self.nodes[i][1] % 8 for i in cone if self.nodes[i][0] == "input"}
            if expanded == closure:
                break
            closure = expanded
        return {
            "and_nodes": sum(self.nodes[i][0] == "AND" for i in reachable),
            "xor_nodes": sum(self.nodes[i][0] == "XOR" for i in reachable),
            "input_variables": len(used_inputs),
            "input_channels": sorted({j % 8 for j in used_inputs}),
            "visible_recurrent_channel_closure": sorted(closure),
            "visible_recurrent_core_binary_nodes": sum(self.nodes[i][0] in ("AND", "XOR") for i in cone),
            "output_expressions": [self.expression(o) for o in self.outputs],
        }


def exact_local_disagreement(left, right):
    """Enumerate the union of relevant patch bits, integrating unused bits out.

    This is a uniform local-state diagnostic, not a distribution of trained
    recurrent states or an exhaustive enumeration of complete CA grid states.
    """
    variables = sorted({dag.nodes[i][1] for dag in (left, right)
                        for i in dag.reachable() if dag.nodes[i][0] == "input"})
    assert len(variables) <= 20, "revisit enumeration budget for a larger rule"
    patch = np.zeros((2**len(variables), 72), dtype=np.uint8)
    patch[:, variables] = (np.arange(len(patch), dtype=np.uint32)[:, None] >> np.arange(len(variables))) & 1
    delta = left.evaluate(patch.reshape(-1, 9, 8)) != right.evaluate(patch.reshape(-1, 9, 8))
    return {"relevant_input_bits": len(variables), "enumerated_assignments": len(patch),
            "any_channel_disagreement_fraction": float(delta.any(-1).mean()),
            "channel_disagreement_fractions": delta.mean(0).tolist()}


def build_dag(gates, wires):
    dag = BooleanDAG()
    inputs = (2 * np.arange(1, 73)).reshape(9, 8)
    x = np.broadcast_to(inputs.T[None], (16, 8, 9)).copy()
    for layer in range(3):
        a, b = wires[layer]
        y = np.empty((16, 8, len(a)), dtype=np.int64)
        for k in range(16):
            for c in range(8):
                for j in range(len(a)):
                    y[k, c, j] = dag.gate(int(gates[layer][k, j]), int(x[k, c, a[j]]), int(x[k, c, b[j]]))
        x = y
    x = np.concatenate([inputs[4], x.transpose(1, 2, 0).reshape(-1)])
    for layer in range(3, 19):
        a, b = wires[layer]
        x = np.array([dag.gate(int(g), int(x[i]), int(x[j])) for g, i, j in zip(gates[layer], a, b)])
    dag.outputs = x.tolist()
    return dag


def validate_algebra():
    # Exhaust all 16 functions, both Boolean inputs, and aliases/complements.
    patch = np.zeros((4, 9, 8), dtype=np.uint8)
    patch[:, 0, 0] = [0, 0, 1, 1]
    patch[:, 0, 1] = [0, 1, 0, 1]
    literals = [0, 1, 2, 3, 4, 5]
    for g in range(16):
        for a in literals:
            for b in literals:
                dag = BooleanDAG()
                dag.outputs = [dag.gate(g, a, b)]
                def read(lit):
                    return (np.zeros(4, np.uint8) if lit // 2 == 0 else patch[:, 0, lit // 2 - 1]) ^ (lit & 1)
                expected = boolean_gate(g, read(a), read(b))
                actual = np.broadcast_to(dag.evaluate(patch)[..., 0], (4,))
                assert np.array_equal(actual, expected), (g, a, b)


def rollout(grids, step, ticks=20):
    states = [grids]
    for _ in range(ticks):
        states.append(step(states[-1]))
    return np.stack(states)


def terminal_metrics(states, target):
    errors = (states[-1, ..., 0] != target[..., 0]).sum(axis=(1, 2))
    return {"hard_bit_errors": int(errors.sum()), "perfect_grid_count": int((errors == 0).sum())}


def flat(arrays, width):
    return np.concatenate([a.reshape(-1, width) for a in arrays])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    manifest = json.loads(args.manifest.read_text())
    verified = verify_bundles(args.bundle_dir, args.artifact_dir, manifest)
    validate_algebra()
    export = load_npz(args.artifact_dir / "analysis/final_hard_circuit.npz")
    wires = [(export[f"wire_{layer}_a"], export[f"wire_{layer}_b"]) for layer in LAYERS]
    evaluation = load_npz(args.artifact_dir / "analysis/fixed_evaluation_set.npz")
    grids, target = evaluation["inputs"].astype(np.uint8), evaluation["target"].astype(np.uint8)
    assert np.array_equal(grids, evaluation["inputs"]) and np.array_equal(target, evaluation["target"])
    expected_evaluations = {v["update"]: v for v in manifest["evaluations"]}
    report = {"run_id": RUN, "analysis_kind": "post-hoc; no training or new initial grids",
              "input_manifest_sha256": sha256(args.manifest.read_bytes()),
              "source_code_commit": manifest["code"]["commit"],
              "bundles": verified, "unique_gate_slots": 3040,
              "boolean_gate_instances_per_cell_tick": 4608,
              "numeric_policy": "FP64 post-hoc softmax/truth-table statistics; uint8 Boolean execution",
              "hardening": "native: argmax logits; truth_round: q >= 0.5, columns 00,01,10,11",
              "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
              "checkpoints": [], "intervals": []}
    previous = None
    all_hard_states = {}
    all_decoded = {}
    previous_dag = None
    for update in range(0, 501, 50):
        checkpoint = load_npz(args.artifact_dir / "checkpoints" / f"checkpoint_update_{update:03d}.npz")
        assert int(checkpoint["update_index"]) == update
        assert int(checkpoint["opt_000"]) == update
        metadata = json.loads(str(checkpoint["metadata_json"]))
        assert metadata["run_id"] == RUN
        assert metadata["code_commit"] == manifest["code"]["commit"]
        assert metadata["config_sha256"] == manifest["config_sha256"]
        params, ps, qs, native, rounded = decode(checkpoint)
        z, p, q = flat(params, 16), flat(ps, 16), flat(qs, 4)
        ids = np.concatenate([g.ravel() for g in native])
        round_ids = np.concatenate([g.ravel() for g in rounded])
        dag = build_dag(native, wires)
        round_dag = build_dag(rounded, wires)
        states = rollout(grids, lambda s: dag.evaluate(patches(s)))
        round_states = rollout(grids, lambda s: round_dag.evaluate(patches(s)))
        probe = load_npz(args.artifact_dir / "analysis" / f"probe_trajectory_update_{update:03d}.npz")
        full_probe = rollout(grids[:1], lambda s: full_step(s, native, wires))
        assert np.array_equal(full_probe, probe["hard"]), f"full replay disagrees at {update}"
        assert np.array_equal(states[:, :1], full_probe), f"simplified replay disagrees at {update}"
        result = terminal_metrics(states, target)
        assert result["hard_bit_errors"] == expected_evaluations[update]["hard_bit_errors"]
        assert result["perfect_grid_count"] == expected_evaluations[update]["perfect_grid_count"]
        if update == 500:
            for layer, g in zip(LAYERS, native):
                assert np.array_equal(g, export[f"gate_{layer}"])
            # Complete full-network replay of all 32 final-checkpoint probes.
            final_full = rollout(grids, lambda s: full_step(s, native, wires))
            assert np.array_equal(states, final_full)
        row = {
            "update": update, "native": result, "truth_round": terminal_metrics(round_states, target),
            "recorded_soft_terminal_summed_squared_error": expected_evaluations[update]["soft_terminal_summed_squared_error"],
            "native_vs_truth_round_gate_disagreements": int((ids != round_ids).sum()),
            "min_truth_round_threshold_distance": float(np.min(np.abs(q - .5))),
            "gate_histogram": np.bincount(ids, minlength=16).tolist(),
            "non_A_slots": int((ids != 3).sum()),
            "non_copy_slots": int(((ids != 3) & (ids != 5)).sum()),
            "mean_gate_entropy_bits": float(-(p * np.log2(p)).sum(-1).mean()),
            "mean_max_gate_probability": float(p.max(-1).mean()),
            "mean_truth_ambiguity": float((4 * q * (1 - q)).mean()),
            "simplified": dag.summary(), "truth_round_simplified": round_dag.summary(),
            "native_vs_truth_round_local_rule": exact_local_disagreement(dag, round_dag),
            "full_probe_trajectory_matches_saved": True,
            "native_visible_errors_by_tick": (states[..., 0] != target[None, ..., 0]).sum(axis=(1, 2, 3)).tolist(),
            "truth_round_visible_errors_by_tick": (round_states[..., 0] != target[None, ..., 0]).sum(axis=(1, 2, 3)).tolist(),
            "native_active_channels_by_tick": states.mean(axis=(1, 2, 3)).tolist(),
            "probe_soft_terminal_visible_mean": float(probe["soft"][-1, ..., 0].mean()),
            "probe_soft_terminal_visible_range": [float(probe["soft"][-1, ..., 0].min()), float(probe["soft"][-1, ..., 0].max())],
            "probe_soft_terminal_channel_means": probe["soft"][-1].mean(axis=(0, 1, 2)).tolist(),
        }
        if previous is not None:
            u0, prev_ids, prev_p, prev_q, prev_states, prev_expr = previous
            report["intervals"].append({
                "from_update": u0, "to_update": update,
                "native_gate_flips": int((ids != prev_ids).sum()),
                "gate_probability_rms_change": float(np.sqrt(np.mean((p - prev_p)**2))),
                "truth_entry_rms_change": float(np.sqrt(np.mean((q - prev_q)**2))),
                "trajectory_all_channel_bit_changes": int((states != prev_states).sum()),
                "trajectory_visible_bit_changes": int((states[..., 0] != prev_states[..., 0]).sum()),
                "terminal_all_channel_bit_changes": int((states[-1] != prev_states[-1]).sum()),
                "terminal_visible_bit_changes": int((states[-1, ..., 0] != prev_states[-1, ..., 0]).sum()),
                "identical_simplified_local_expressions": row["simplified"]["output_expressions"] == prev_expr,
                "exact_local_rule_disagreement": exact_local_disagreement(previous_dag, dag),
            })
        previous = (update, ids, p, q, states, row["simplified"]["output_expressions"])
        previous_dag = dag
        all_hard_states[update] = states
        all_decoded[update] = (ids, row["simplified"]["output_expressions"])
        report["checkpoints"].append(row)
        print(json.dumps({"update": update, "native": row["native"], "truth_round": row["truth_round"],
                          "non_A_slots": row["non_A_slots"], "simplified": row["simplified"]}), flush=True)
    s250, s500 = all_hard_states[250], all_hard_states[500]
    report["update250_to_500"] = {
        "native_gate_endpoint_differences": int((all_decoded[250][0] != all_decoded[500][0]).sum()),
        "native_gate_flips_over_saved_intervals": sum(i["native_gate_flips"] for i in report["intervals"] if i["from_update"] >= 250),
        "trajectory_all_channel_bit_changes": int((s250 != s500).sum()),
        "trajectory_visible_bit_changes": int((s250[..., 0] != s500[..., 0]).sum()),
        "terminal_all_channel_bit_changes": int((s250[-1] != s500[-1]).sum()),
        "terminal_visible_bit_changes": int((s250[-1, ..., 0] != s500[-1, ..., 0]).sum()),
    }
    report["validation"] = {"bundle_sha256_matches": True, "constituent_sha256_matches_original_manifest": True,
                            "boolean_algebra_exhaustive_gate_checks": 576,
                            "full_replay_matches_saved_21_tick_probe_at_all_11_checkpoints": True,
                            "all_32_terminal_errors_match_original_manifest_at_all_11_checkpoints": True,
                            "final_export_ids_match_checkpoint": True,
                            "full_final_network_matches_simplified_for_all_32_trajectories": True}
    report["analysis_seconds"] = time.perf_counter() - start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"validation": report["validation"], "seconds": report["analysis_seconds"],
                      "update250_to_500": report["update250_to_500"]}), flush=True)


if __name__ == "__main__":
    main()
