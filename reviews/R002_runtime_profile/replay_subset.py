#!/usr/bin/env python3
"""Replay the six available source circuits and audit recorded orbit summaries."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reviews/R002_circuits"))
import analyze_subset as a
import refine_circuits as refine

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    result_path = ROOT / "results/R002_formal_attempt01_runtime_profile_attempt01"
    modes = json.loads((result_path / "runs.json").read_text())["modes"]
    manifest = json.loads((result_path / "manifest.json").read_text())
    with gzip.open(result_path / "trajectories.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    probe_path = args.artifact_root / "reconstructed_fixed_evaluation_set.npz"
    assert hashlib.sha256(probe_path.read_bytes()).hexdigest() == manifest["original_probe"]["sha256"]
    probe = a.load(probe_path)
    target = probe["target"][0,...,0].astype(np.uint8)
    fresh = np.random.Generator(np.random.PCG64(20260910)).integers(0,2,size=(32,16,16,8),dtype=np.uint8)
    assert hashlib.sha256(fresh.tobytes()).hexdigest() == manifest["fresh_probe"]["raw_array_sha256"]
    inputs = np.concatenate([probe["inputs"].astype(np.uint8),fresh,np.zeros((1,16,16,8),np.uint8),np.ones((1,16,16,8),np.uint8)])
    ids = [f"original_probe_{i:02d}" for i in range(32)] + [f"fresh_probe_{i:02d}" for i in range(32)] + ["all_zero","all_one"]
    seen = {}
    verified = []
    for mode in modes:
        if mode["run_id"] not in a.NAMES:
            continue
        for reference in (mode["source"]["checkpoint"],mode["source"]["export"]):
            path = args.artifact_root / mode["run_id"] / Path(reference["path"]).name
            data = path.read_bytes()
            assert len(data) == reference["bytes"] and hashlib.sha256(data).hexdigest() == reference["sha256"]
        export = a.load(args.artifact_root / mode["run_id"] / f"final_{mode['hardening']}_circuit.npz")
        raw_gates = [export[f"gate_{label}"] for label in a.LABELS]
        assert all(np.issubdtype(g.dtype,np.integer) and np.all((g>=0)&(g<=15)) for g in raw_gates)
        # The profiler hashes canonical uint8 IDs; the exported arrays are int32.
        gates = [g.astype(np.uint8) for g in raw_gates]
        wires = [(export[f"wire_{label}_a"],export[f"wire_{label}_b"]) for label in a.LABELS]
        ck = a.load(args.artifact_root / mode["run_id"] / "checkpoint_update_500.npz")
        _,decoded = a.decode(ck,mode["representation"])
        assert all(np.array_equal(g,h) for g,h in zip(gates,decoded[mode["hardening"]]))
        rule_hash = a.tree_digest(gates + [v for pair in wires for v in pair])
        assert rule_hash == mode["source"]["rule_sha256"]
        if rule_hash not in seen:
            d = refine.build(gates,wires)
            states = a.hard_rollout(inputs,d,256)
            full_first = a.base.rollout(inputs[:1],lambda s:a.base.full_step(s,gates,wires),20)
            assert np.array_equal(states[:21,:1],full_first)
            errors = (states[...,0] != target).sum(axis=(2,3))
            cert = a.abstract_certificate(d,ticks=256)
            summaries = []
            for i in range(66):
                # Use NumPy's exact unique rows instead of the producer's seen-state dictionary.
                packed = np.packbits(states[:,i].reshape(257,-1),axis=1,bitorder="big")
                keys = np.ascontiguousarray(packed).view(np.dtype((np.void,256))).reshape(-1)
                _,first,inverse = np.unique(keys,return_index=True,return_inverse=True)
                repeats = np.flatnonzero(first[inverse] < np.arange(257))
                mu = int(first[inverse[repeats[0]]]) if len(repeats) else None
                period = int(repeats[0])-mu if len(repeats) else None
                e = errors[:,i]
                hits = np.flatnonzero(e==0)
                wrong = np.flatnonzero(e!=0)
                suffix = (int(wrong[-1])+1 if len(wrong) else 0) if e[-1]==0 else None
                s = {"cycle_entry_tick":mu,"cycle_period":period,"errors_at_20":int(e[20]),
                     "first_hit_tick":int(hits[0]) if len(hits) else None,
                     "observed_correct_suffix_start":suffix,
                     "observed_correct_suffix_length":257-suffix if suffix is not None else 0}
                if mu is not None:
                    c = e[mu:mu+period]
                    s.update({"correct_phase_indices":np.flatnonzero(c==0).tolist(),
                              "cycle_target_fraction":float(np.mean(c==0)),
                              "cycle_min_errors":int(c.min()),"cycle_mean_errors":float(c.mean()),"cycle_max_errors":int(c.max())})
                    cyc = [row.tobytes() for row in packed[mu:mu+period]]
                    start = min(range(period),key=cyc.__getitem__)
                    s["cycle_signature_sha256"] = hashlib.sha256(b"".join(cyc[start:]+cyc[:start])).hexdigest()
                summaries.append(s)
            seen[rule_hash] = (summaries,errors,cert)
        summaries,errors,cert = seen[rule_hash]
        stored = {row["initialization_id"]:row for row in rows if row["run_id"]==mode["run_id"] and row["hardening"]==mode["hardening"]}
        for i,identifier in enumerate(ids):
            for key,value in summaries[i].items():
                assert stored[identifier][key] == value,(mode["run_id"],mode["hardening"],identifier,key)
        for group,indices in {"original_probe":slice(0,32),"fresh_probe":slice(32,64),"all_zero":slice(64,65),"all_one":slice(65,66)}.items():
            for tick,values in mode["initialization_groups"][group]["report_ticks"].items():
                ee = errors[int(tick),indices]
                assert int(ee.sum()) == values["aggregate_errors"]
                assert int((ee==0).sum()) == values["perfect_initializations"]
        status = mode["universal_state_certificate"]
        assert cert["abstract_fixed_point_tick"] == status["invariant_abstraction_tick"]
        assert cert["first_all_visible_correct"] == status["certified_visible_settling_tick"]
        verified.append({"run_id":mode["run_id"],"hardening":mode["hardening"],"trajectories":66,"ticks":257,"source_hashes_verified":True,"recorded_orbit_statistics_match":True})
        print(mode["run_id"],mode["hardening"],"verified",flush=True)
    assert len(verified)==12
    report = {"source_commit":"60ef04441ff16e26e395e28b9c5c63cb7d2849af","source_artifact_root":str(args.artifact_root),
              "scope":"Six available training runs, both hardening modes; 792 logical trajectories. No replay claim is made for the other 58 source circuits.",
              "numpy":np.__version__,"unique_rules_replayed":len(seen),"modes":verified}
    (OUT / "subset_replay.json").write_text(json.dumps(report,indent=2)+"\n")

if __name__ == "__main__":
    main()
