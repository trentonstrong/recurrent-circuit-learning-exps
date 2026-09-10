#!/usr/bin/env python3
"""Independent review of R002's committed manifests and metrics. No training."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

CONDITIONS = (
    "categorical_reference_decay", "truth_reference_decay",
    "categorical_no_decay", "truth_no_decay",
)
EXPECTED_CONFIG = "9632f2135e8262903c40cdaeba211e4f5bebac22c2b8f45c0e2100b993111bb9"
EXPECTED_PROBE = "d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f"
EXPECTED_LOCK = "7a8e627cd9851d7426502bc3734bcedcfe48277575aec363877d4e7b717b2125"
EMPTY_HASH = hashlib.sha256(b"").hexdigest()

def summary(a):
    a = np.asarray(a, dtype=float)
    return {"median": float(np.median(a)), "mean": float(np.mean(a)),
            "min": float(np.min(a)), "max": float(np.max(a)),
            "q25": float(np.quantile(a, .25)), "q75": float(np.quantile(a, .75))}

def wilson(k, n):
    z = 1.959963984540054
    den = 1 + z*z/n
    center = (k/n + z*z/(2*n))/den
    half = z * math.sqrt(k/n*(1-k/n)/n + z*z/(4*n*n))/den
    return [max(0., center-half), min(1., center+half)]

def exact_mcnemar(a, b):
    aw = int(np.sum(a & ~b))
    bw = int(np.sum(b & ~a))
    n = aw + bw
    p = 1. if n == 0 else min(1., 2*sum(math.comb(n,k) for k in range(min(aw,bw)+1))/2**n)
    return {"first_only_successes": aw, "second_only_successes": bw, "two_sided_p": p}

def load_records(root, out):
    identity = json.loads((out/"source_inputs.json").read_text())
    for item in identity["inputs"]:
        raw = (root/item["path"]).read_bytes()
        git_sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        assert len(raw) == item["bytes"], item["path"]
        assert git_sha == item["git_blob_sha"], item["path"]
    records = {}
    all_artifacts = []
    for seed in range(16):
        for condition in CONDITIONS:
            run_id = f"R002_formal_attempt01_seed{seed:02d}_{condition}"
            path = root/"results"/run_id
            m = json.loads((path/"manifest.json").read_text())
            metrics = [json.loads(s) for s in (path/"metrics.jsonl").read_text().splitlines()]
            assert m["run_id"] == run_id
            assert (m["seed"],m["condition"],m["status"],m["current_update"]) == (seed,condition,"completed",500)
            assert m["failure_reason"] is None
            assert m["code"]["commit"] == identity["training_commit"]
            if m["code"]["dirty"]:
                assert m["code"]["tracked_patch_sha256"] == EMPTY_HASH
                assert all(s.startswith("?? results/R002_formal_attempt01_seed") for s in m["code"]["status"])
            else:
                assert not m["code"].get("status")
            assert m["config_sha256"] == EXPECTED_CONFIG
            assert m["probe"]["sha256"] == EXPECTED_PROBE
            assert m["environment"]["lock_sha256"] == EXPECTED_LOCK
            assert m["environment"]["jax_enable_x64"] is False
            assert m["environment"]["jax_threefry_partitionable"] is False
            assert m["kernel"] == "common_multilinear_truth_table"
            assert m["pairing"]["initial_effective_table_discrepancy"]["ok"]
            assert [x["update"] for x in metrics] == list(range(1,501))
            assert [x["update"] for x in m["evaluations"]] == list(range(0,501,50))
            assert [x["update"] for x in m["checkpoints"]] == list(range(0,501,50))
            assert metrics[-1] == m["final_training_metric"]
            assert all(np.isfinite(float(v)) for x in metrics for v in x.values())
            for e in m["evaluations"]:
                assert np.isfinite(e["soft_terminal_summed_squared_error"])
                for hard in ("native","common"):
                    assert e[hard]["exact_all_32"] == (e[hard]["bit_errors"] == 0)
                    assert e[hard]["terminal_summed_squared_error"] == e[hard]["bit_errors"]
                    assert abs(e[f"{hard}_soft_hard_gap"] - (
                        e[hard]["bit_errors"] - e["soft_terminal_summed_squared_error"])) < 1e-8
                if condition.startswith("truth"):
                    assert e["native"] == e["common"]
            for hard in ("native","common"):
                actual = next((e["update"] for e in m["evaluations"] if e[hard]["exact_all_32"]), None)
                assert m["first_exact_saved_checkpoint"][hard] == actual
            artifacts = m["artifacts"] + [a for c in m["checkpoints"] for a in (c["checkpoint"], c["probe_trajectory"])]
            assert len(artifacts) == 24
            assert all(a["bytes"] > 0 and len(a["sha256"]) == 64 for a in artifacts)
            all_artifacts += artifacts
            records[(seed,condition)] = (m,metrics)
        pair = [records[(seed,c)][0]["pairing"] for c in CONDITIONS]
        for key in ("wiring_sha256", "categorical_initial_params_sha256",
                    "truth_initial_params_sha256", "training_stream_sha256"):
            assert len({p[key] for p in pair}) == 1, (seed,key)
    assert len({a["path"] for a in all_artifacts}) == len(all_artifacts)
    return records, {
        "source_commit": identity["reviewed_commit"],
        "training_commit": identity["training_commit"],
        "git_blobs_verified": len(identity["inputs"]),
        "completed_runs": len(records),
        "ordered_update_records": sum(len(v[1]) for v in records.values()),
        "evaluations": sum(len(v[0]["evaluations"]) for v in records.values()),
        "paired_seed_groups": 16,
        "artifact_records": len(all_artifacts),
        "artifact_bytes_from_records": sum(a["bytes"] for a in all_artifacts),
        "artifact_payloads_independently_verified": False,
        "artifact_note": "Checkpoint/trajectory/circuit payloads are not in git and no R002 release was available during review.",
        "clipped_updates": sum(x["clipped_element_count"] > 0 for _,ms in records.values() for x in ms),
        "largest_gradient_entry": max(x["gradient_max_abs"] for _,ms in records.values() for x in ms),
        "clean_runs": sum(not m["code"]["dirty"] for m,_ in records.values()),
        "dirty_run_tracked_code_patch_hashes": sorted({m["code"]["tracked_patch_sha256"] for m,_ in records.values() if m["code"]["dirty"]}),
    }

def analyze(root, out):
    records, integrity = load_records(root,out)
    rows, trajectories, condition_stats = [], {}, {}
    for condition in CONDITIONS:
        paths, flips, endpoints, first_changes, softs, hards = [], [], [], [], [], []
        for seed in range(16):
            m, ms = records[(seed,condition)]
            es = m["evaluations"]
            final = es[-1]
            q_path = np.cumsum([x["q_step_l2"] for x in ms])
            common_flips = np.cumsum([x["common_gate_id_changes"] for x in ms])
            native_flips = np.cumsum([x["native_gate_id_changes"] for x in ms])
            layer_summaries = [x for layers in final["gate_diagnostics"]["layers"].values() for x in layers]
            entropy = sum(x["gate_count"]*x["mean_truth_bit_entropy"] for x in layer_summaries)/sum(x["gate_count"] for x in layer_summaries)
            first_change = next((x["update"] for x in ms if x["common_gate_id_changes"]), None)
            first_changes.append(first_change if first_change is not None else 501)
            paths.append(q_path)
            flips.append(common_flips)
            endpoints.append(final["effective_table_displacement"]["l2"])
            softs.append([e["soft_terminal_summed_squared_error"] for e in es])
            hards.append([e["common"]["bit_errors"] for e in es])
            row = {
                "seed":seed, "condition":condition,
                "final_soft_sse":final["soft_terminal_summed_squared_error"],
                "final_common_errors":final["common"]["bit_errors"],
                "final_native_errors":final["native"]["bit_errors"],
                "common_success":final["common"]["exact_all_32"],
                "native_success":final["native"]["exact_all_32"],
                "first_exact_common":m["first_exact_saved_checkpoint"]["common"],
                "ever_exact_common":any(e["common"]["exact_all_32"] for e in es),
                "first_common_gate_change":first_change,
                "common_gate_change_events":int(common_flips[-1]),
                "native_gate_change_events":int(native_flips[-1]),
                "updates_with_common_gate_change":sum(x["common_gate_id_changes"]>0 for x in ms),
                "q_path_length":float(q_path[-1]),
                "q_endpoint_l2":float(endpoints[-1]),
                "q_first_step_l2":ms[0]["q_step_l2"],
                "q_last_50_steps_median_l2":float(np.median([x["q_step_l2"] for x in ms[-50:]])),
                "final_mean_truth_bit_entropy":float(entropy),
                "final_min_rounding_margin":min(x["minimum_rounding_margin"] for x in layer_summaries),
                "soft_sse_change_400_to_500":softs[-1][-1]-softs[-1][-3],
                "common_error_change_400_to_500":hards[-1][-1]-hards[-1][-3],
                "last_100_common_gate_change_events":sum(x["common_gate_id_changes"] for x in ms[-100:]),
                "last_100_q_path_length":float(sum(x["q_step_l2"] for x in ms[-100:])),
                "training_final_50_median_loss":float(np.median([x["pre_update_soft_loss"] for x in ms[-50:]])),
                "synchronized_update_seconds":m["synchronized_update_seconds"],
            }
            rows.append(row)
        cr = [r for r in rows if r["condition"]==condition]
        success = sum(r["common_success"] for r in cr)
        condition_stats[condition] = {
            "successes":success, "success_seeds":[r["seed"] for r in cr if r["common_success"]],
            "wilson_95":wilson(success,16),
            **{key:summary([r[key] for r in cr]) for key in (
                "final_soft_sse", "final_common_errors", "final_native_errors",
                "q_path_length", "q_endpoint_l2", "q_first_step_l2",
                "common_gate_change_events", "native_gate_change_events",
                "final_mean_truth_bit_entropy", "final_min_rounding_margin",
                "soft_sse_change_400_to_500", "common_error_change_400_to_500",
                "last_100_common_gate_change_events", "last_100_q_path_length",
                "synchronized_update_seconds")},
            "first_common_gate_change":summary(first_changes),
            "soft_improves_last_100":sum(r["soft_sse_change_400_to_500"]<0 for r in cr),
            "hard_improves_last_100":sum(r["common_error_change_400_to_500"]<0 for r in cr),
            "hard_worsens_last_100":sum(r["common_error_change_400_to_500"]>0 for r in cr),
        }
        trajectories[condition] = {
            "saved_updates":list(range(0,501,50)),
            "soft_sse":np.asarray(softs).tolist(), "common_errors":np.asarray(hards).tolist(),
            "q_path_length":np.asarray(paths).tolist(),
            "common_gate_change_events":np.asarray(flips).tolist(),
        }
    rng = np.random.default_rng(20260910)
    samples = rng.integers(0,16,size=(100000,16))
    signs = 2*((np.arange(65536)[:,None] >> np.arange(16)) & 1)-1
    by = {(r["seed"],r["condition"]):r for r in rows}
    contrasts = {}
    for first,second in (
        (CONDITIONS[1],CONDITIONS[0]), (CONDITIONS[3],CONDITIONS[2]),
        (CONDITIONS[2],CONDITIONS[0]), (CONDITIONS[3],CONDITIONS[1])):
        a = [by[(s,first)] for s in range(16)]
        b = [by[(s,second)] for s in range(16)]
        sa = np.array([r["common_success"] for r in a])
        sb = np.array([r["common_success"] for r in b])
        result = {"first":first,"second":second,
                  "success_rate_difference":float(np.mean(sa.astype(int)-sb.astype(int))),
                  "mcnemar":exact_mcnemar(sa,sb)}
        for key in ("final_common_errors","final_soft_sse","q_path_length",
                    "common_gate_change_events","q_first_step_l2"):
            delta = np.array([x[key]-y[key] for x,y in zip(a,b)])
            mean = float(np.mean(delta))
            result[key] = {
                "mean_difference":mean, "median_difference":float(np.median(delta)),
                "first_lower":int(np.sum(delta<0)), "first_higher":int(np.sum(delta>0)),
                "equal":int(np.sum(delta==0)),
                "bootstrap_mean_95":np.quantile(delta[samples].mean(1),[.025,.975]).tolist(),
                "exploratory_exact_sign_flip_p":float(np.mean(abs(signs@delta/16)>=abs(mean)-1e-12)),
            }
        contrasts[first+"_minus_"+second] = result
    note = {
        "bootstrap": "100000 paired seed resamples; NumPy default_rng(20260910); same indices across contrasts. This independently seeded implementation need not bit-match the partner's Python-random bootstrap endpoints.",
        "sign_flip": "Exploratory two-sided exact within-pair sign-flip test of mean differences (65536 sign assignments); no multiplicity correction and no confirmatory causal claim.",
        "gate_changes": "Counts are per-slot ID change events, including repeated flips; not counts of distinct functions, neutral paths, or compressed descriptions.",
        "q_path": "Sum of all per-update Euclidean displacements of the 12160 effective truth coordinates; coordinate-comparable but not a behavior-space distance.",
        "entropy": "Gate-count-weighted mean Bernoulli entropy of effective truth entries; not categorical entropy and not description length.",
    }
    report = {"integrity":integrity,"methods":note,"conditions":condition_stats,
              "paired_contrasts":contrasts,"runs":rows}
    (out/"analysis.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (out/"plot_data.json").write_text(json.dumps(trajectories,allow_nan=False)+"\n")
    printed = {
        "integrity":integrity,
        "conditions":{c:{k:v if not isinstance(v,dict) else v.get("median",v)
                           for k,v in stats.items()} for c,stats in condition_stats.items()},
        "paired_contrasts":contrasts,
        "lost_exact_solutions":[r for r in rows if r["ever_exact_common"] and not r["common_success"]],
        "successful_runs":[r for r in rows if r["common_success"]],
    }
    print(json.dumps(printed,indent=2,allow_nan=False))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    out = args.repo/"reviews"/"R002_formal_attempt01"
    out.mkdir(parents=True,exist_ok=True)
    analyze(args.repo,out)
