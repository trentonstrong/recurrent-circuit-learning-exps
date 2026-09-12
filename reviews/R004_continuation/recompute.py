#!/usr/bin/env python3
"""Independently audit R004's released records; no JAX training is replayed.

Usage: python recompute.py --bundle EXTRACTED_REVIEW --reference FE14_CHECKOUT
                          --output REVIEW_DIRECTORY
The bundle's CONTENTS.json authenticates its payloads against the verified
release archive; reference is the pinned fe14ca6 repository checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CONDITIONS = ["categorical_reference_decay", "truth_reference_decay",
              "categorical_no_decay", "truth_no_decay"]
ANALYSIS = "R004_formal_attempt01_analysis_attempt03"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(path, record):
    assert Path(path).stat().st_size == record["bytes"], str(path)
    assert digest(path) == record["sha256"], str(path)


def small(record):
    return {"min": int(np.min(record)), "median": float(np.median(record)),
            "mean": float(np.mean(record)), "max": int(np.max(record))}


def audit(bundle, reference):
    inventory = read(bundle / "CONTENTS.json")
    for record in inventory["files"]:
        verify(bundle / record["path"], record)
    config_path = bundle / "implementation/configs/experiments/r004_training_continuation_v1.json"
    config_hash = digest(config_path)
    assert config_hash == "9e6aed592c8214df37486ceaae20af1061d0498b4784be1a5c7380afde87907a"
    assert config_path.read_bytes() == (reference / "configs/experiments/r004_training_continuation_v1.json").read_bytes()
    config = read(config_path)
    lock = bundle / "implementation/envs/jax-gpu/uv.lock"
    assert digest(lock) == config["runtime"]["expected_lock_sha256"]
    formal = read(bundle / "results/R004_formal_attempt01/manifest.json")
    assert formal["status"] == "completed" and formal["completed_runs"] == 64
    assert formal["failures"] == []
    result = read(bundle / "results" / ANALYSIS / "analysis.json")
    structure = [json.loads(s) for s in (bundle / "results" / ANALYSIS / "structure.jsonl").read_text().splitlines()]
    assert len(structure) == 896
    struct = {(r["seed"], r["condition"], r["global_update"], r["hardening"]): r for r in structure}
    assert len(struct) == 896
    assert set(struct) == {(s,c,k,h) for s in range(16) for c in CONDITIONS
                           for k in range(500,2001,250) for h in ("common","native")}
    manifests = {}
    parents = {}
    series = {}
    metric_stats = {}
    count_updates = count_checkpoints = count_exports = 0
    matching_train_sources = Counter()
    analysis_hash_versions = Counter()
    identities = []
    digests_by_seed = defaultdict(list)
    for path in sorted((bundle / "results").glob("R004_formal_attempt01_seed*/manifest.json")):
        m = read(path)
        s,c = m["seed"],m["condition"]
        assert (s,c) not in manifests
        manifests[s,c] = m
        assert m["status"] == "completed" and m["current_update"] == 2000
        assert m["config_sha256"] == config_hash
        assert [e["global_update"] for e in m["evaluations"]] == list(range(500,2001,50))
        assert [e["update"] for e in m["checkpoints"]] == list(range(550,2001,50))
        assert {(e["global_update"],e["hardening"]) for e in m["structural_exports"]} == {(k,h) for k in range(500,2001,250) for h in ("native","common")}
        parent = reference / "results" / m["lineage"]["parent_run_id"] / "manifest.json"
        verify(parent,m["lineage"]["parent_manifest"])
        p = read(parent); parents[s,c] = p
        parent_checkpoint = [e for e in p["checkpoints"] if e["update"] == 500][0]
        assert parent_checkpoint["checkpoint"]["sha256"] == m["lineage"]["parent_checkpoint_sha256"]
        assert p["pairing"]["wiring_sha256"] == m["lineage"]["wiring_sha256"]
        assert m["parent_baseline"] == p["evaluations"][-1]
        es = [{**e,"global_update":e["update"]} for e in p["evaluations"][:-1]] + m["evaluations"]
        assert [e["global_update"] for e in es] == list(range(0,2001,50))
        series[s,c] = es
        for e in es:
            for h in ("common","native"):
                assert e[h]["exact_all_32"] == (e[h]["bit_errors"] == 0)
                assert e[h]["exact_all_32"] == (e[h]["perfect_grid_count"] == 32)
        metric_path = path.parent / "metrics.jsonl"
        verify(metric_path,m["metrics"])
        rows = [json.loads(line) for line in metric_path.read_text().splitlines()]
        assert [r["global_update"] for r in rows] == list(range(501,2001))
        assert [r["continuation_update"] for r in rows] == list(range(1,1501))
        indexed = "".join(f'{r["global_update"]}:{r["training_batch_sha256"]}\n' for r in rows)
        batch_digest = hashlib.sha256(indexed.encode("ascii")).hexdigest()
        assert batch_digest == m["continuation_batch_hashes_sha256"]
        digests_by_seed[s].append(batch_digest)
        count_updates += len(rows); count_checkpoints += len(m["checkpoints"])
        count_exports += sum(e["global_update"] > 500 for e in m["structural_exports"])
        for rel in ("recurrent_circuit_learning/r004_runner.py","scripts/r004.py"):
            verify(bundle / "implementation" / rel,m["code"]["implementation_sources"][rel])
            matching_train_sources[rel] += 1
        analysis_hash_versions[m["code"]["implementation_sources"]["recurrent_circuit_learning/r004_analysis.py"]["sha256"]] += 1
        metric_stats[s,c] = {
            "synchronized_seconds": float(sum(r["synchronized_seconds"] for r in rows)),
            "wall_seconds": m["training_loop_wall_seconds"],
            "common_changes": int(sum(r["common_gate_id_changes"] for r in rows)),
            "native_changes": int(sum(r["native_gate_id_changes"] for r in rows)),
            "last_250_common_changes": int(sum(r["common_gate_id_changes"] for r in rows[-250:])),
            "last_250_mean_gradient_l2": float(np.mean([r["gradient_l2"] for r in rows[-250:]])),
        }
        assert math.isclose(metric_stats[s,c]["synchronized_seconds"],m["synchronized_update_seconds"],rel_tol=1e-12)
        identities.append({"seed":s,"condition":c,"manifest_sha256":digest(path),"parent_manifest_sha256":digest(parent),"metrics_sha256":digest(metric_path),"batch_digest":batch_digest})
    assert set(manifests) == {(s,c) for s in range(16) for c in CONDITIONS}
    assert all(len(ds)==4 and len(set(ds))==1 for ds in digests_by_seed.values())
    assert (count_updates,count_checkpoints,count_exports) == (96000,1920,768)

    computed_conditions = {}
    detail = []
    for c in CONDITIONS:
        successes_before=[]; successes_after=[]; ever=[]; fleeting=[]; relapses=[]
        common_trajectory=[]; native_trajectory=[]
        before_errors=[]; after_errors=[]; soft_change=[]; sizes={}
        for j,k in enumerate(range(0,2001,50)):
            common_trajectory.append(sum(series[s,c][j]["common"]["exact_all_32"] for s in range(16)))
            native_trajectory.append(sum(series[s,c][j]["native"]["exact_all_32"] for s in range(16)))
        for k in range(500,2001,250):
            values=[struct[s,c,k,"common"]["visible_core_binary_nodes"] for s in range(16)]
            sizes[k]={**small(values),"raw_seed_values":values}
        for s in range(16):
            es=series[s,c]; byk={e["global_update"]:e for e in es}
            e0,e1=byk[500],byk[2000]
            status=[e["common"]["exact_all_32"] for e in es]
            before=e0["common"]["exact_all_32"];after=status[-1]
            if before: successes_before.append(s)
            if after: successes_after.append(s)
            if any(status):ever.append(s)
            if any(status) and not after:fleeting.append(s)
            lost_at=[es[j]["global_update"] for j in range(1,len(es)) if status[j-1] and not status[j]]
            if lost_at:relapses.append(s)
            before_errors.append(e0["common"]["bit_errors"]);after_errors.append(e1["common"]["bit_errors"])
            soft_change.append(e1["soft_terminal_summed_squared_error"]-e0["soft_terminal_summed_squared_error"])
            first=next((e["global_update"] for e in es if e["common"]["exact_all_32"]),None)
            item={"seed":s,"condition":c,"common_errors_500":before_errors[-1],"common_errors_2000":after_errors[-1],"first_observed_exact_update":first,"lost_success_at_saved_updates":lost_at,
                  "common_sizes":{k:struct[s,c,k,"common"]["visible_core_binary_nodes"] for k in range(500,2001,250)},
                  "soft_losses":{k:byk[k]["soft_terminal_summed_squared_error"] for k in (500,1000,1500,2000)},
                  "soft_loss_change":soft_change[-1],**metric_stats[s,c]}
            detail.append(item)
            reported=next(r for r in result["endpoint_rows"] if r["seed"]==s and r["condition"]==c)
            for name in ("common_errors_500","common_errors_2000","first_observed_exact_update"):
                assert reported[name] == item[name],(s,c,name)
            assert reported["success_timeline"] == [{"global_update":e["global_update"],"common_exact_all_32":e["common"]["exact_all_32"]} for e in es]
        old,new=set(successes_before),set(successes_after)
        summary={"common_exact_500":len(old),"common_exact_2000":len(new),"native_exact_2000":native_trajectory[-1],
                 "history":{"ever_observed_exact":len(ever),"final_inexact_after_observed_success":len(fleeting),"never_observed_exact":16-len(ever)},
                 "transitions":{"gained_at_endpoint":len(new-old),"lost_at_endpoint":len(old-new),"exact_at_both_endpoints":len(old&new),"inexact_at_both_endpoints":16-len(old|new)}}
        assert summary == result["condition_summary"][c]
        computed_conditions[c]={**summary,"success_seeds_500":successes_before,"success_seeds_2000":successes_after,
          "ever_success_seeds":ever,"transient_success_then_failed_endpoint_seeds":fleeting,"any_saved_history_relapse_seeds":relapses,
          "updates":list(range(0,2001,50)),"common_success_trajectory":common_trajectory,"native_success_trajectory":native_trajectory,
          "common_sizes":sizes,"hard_error_500":small(before_errors),"hard_error_2000":small(after_errors),
          "soft_loss_improved_count":int(np.sum(np.array(soft_change)<0)),"soft_loss_changes":soft_change,
          "median_synchronized_seconds_per_update":float(np.median([metric_stats[s,c]["synchronized_seconds"]/1500 for s in range(16)])),
          "common_changes_total":sum(metric_stats[s,c]["common_changes"] for s in range(16))}

    hardening_gap={}
    for c in CONDITIONS:
        rs=[r for r in detail if r["condition"]==c]
        certified=[r for r in rs if r["soft_losses"][2000]<0.25]
        hardening_gap[c]={
           "sufficient_bound":"sum_i (soft_output_i-target_i)^2 < 1/4 implies every output is within 1/2 of its target; sufficient, not necessary",
           "output_count":8192,
           "certified_soft_output_threshold_success_count":len(certified),
           "certified_soft_output_threshold_success_seeds":[r["seed"] for r in certified],
           "certified_soft_readout_but_common_hard_failed_seeds":[r["seed"] for r in certified if r["common_errors_2000"]>0],
           "largest_certifying_sse":max((r["soft_losses"][2000] for r in certified),default=None),
           "median_sse_500":float(np.median([r["soft_losses"][500] for r in rs])),
           "median_sse_2000":float(np.median([r["soft_losses"][2000] for r in rs])),
           "size_increased_count":sum(r["common_sizes"][2000]>r["common_sizes"][500] for r in rs),
        }

    boot=np.random.Generator(np.random.PCG64(20260911)).integers(0,16,size=(100000,16),dtype=np.int64)
    assert hashlib.sha256(boot.tobytes()).hexdigest() == result["bootstrap_identity"]["indices_sha256"]
    paired={}
    for history in ("reference_decay","no_decay"):
        cat,truth="categorical_"+history,"truth_"+history
        cat_success=set(computed_conditions[cat]["success_seeds_2000"])
        truth_success=set(computed_conditions[truth]["success_seeds_2000"])
        a,b=len(cat_success-truth_success),len(truth_success-cat_success)
        p=min(1.,2*sum(math.comb(a+b,k) for k in range(min(a,b)+1))/(2**(a+b))) if a+b else 1.
        reported=result["endpoint_coordinate_mcnemar"][history]
        assert (a,b,p)==(reported["categorical_only"],reported["truth_only"],reported["p_value_two_sided_exact"])
        comparisons={}
        deltas=[]
        for name in ("paired_hard_error_change_difference","size_gap_change"):
            values=[]
            for s in range(16):
                c0,c1=manifests[s,cat]["evaluations"][0],manifests[s,cat]["evaluations"][-1]
                t0,t1=manifests[s,truth]["evaluations"][0],manifests[s,truth]["evaluations"][-1]
                if name.startswith("paired_hard"):
                    x=(t1["common"]["bit_errors"]-t0["common"]["bit_errors"])-(c1["common"]["bit_errors"]-c0["common"]["bit_errors"])
                else:
                    x=(struct[s,truth,2000,"common"]["visible_core_binary_nodes"]-struct[s,cat,2000,"common"]["visible_core_binary_nodes"])-(struct[s,truth,500,"common"]["visible_core_binary_nodes"]-struct[s,cat,500,"common"]["visible_core_binary_nodes"])
                values.append(x)
            arr=np.array(values,dtype=np.float64)
            out={"raw_seed_values":arr.tolist(),"mean":float(arr.mean()),"percentile_95_interval":np.quantile(arr[boot].mean(1),[.025,.975]).tolist()}
            assert out == result["paired_bootstrap"][history][name]
            comparisons[name]=out
        gaps={k:[struct[s,truth,k,"common"]["visible_core_binary_nodes"]-struct[s,cat,k,"common"]["visible_core_binary_nodes"] for s in range(16)] for k in range(500,2001,250)}
        paired[history]={"categorical_only":a,"truth_only":b,"p":p,"both_success_seeds":sorted(cat_success&truth_success),
          "gap_positive_in_all_pairs_at_all_times":all(x>0 for vs in gaps.values() for x in vs),
          "size_gap_by_update":{k:small(vs) for k,vs in gaps.items()},
          "size_gap_increased_count":sum(gaps[2000][s]>gaps[500][s] for s in range(16)),**comparisons}
    ordered=sorted(paired,key=lambda k:paired[k]["p"])
    adjusted={ordered[0]:min(1,2*paired[ordered[0]]["p"])}
    adjusted[ordered[1]]=max(adjusted[ordered[0]],paired[ordered[1]]["p"])
    for k,v in adjusted.items():
        assert v == result["endpoint_coordinate_mcnemar"][k]["holm_adjusted_p_value"]
        paired[k]["holm_adjusted_p"]=v

    with gzip.open(bundle / "results" / ANALYSIS / "runtime_profiles.jsonl.gz","rt") as f:
        runtime=[json.loads(s) for s in f]
    assert len(runtime)==512
    assert {(r["seed"],r["condition"],r["global_update"],r["hardening"]) for r in runtime} == {(s,c,k,h) for s in range(16) for c in CONDITIONS for k in (500,1000,1500,2000) for h in ("common","native")}
    runtime_groups={}
    for c in CONDITIONS:
        runtime_groups[c]={}
        for k in (500,1000,1500,2000):
            rs=[r for r in runtime if r["condition"]==c and r["global_update"]==k and r["hardening"]=="common"]
            cert=Counter(r["universal_state_certificate"]["status"] for r in rs)
            fully=[r["seed"] for r in rs if all(t["errors_at_20"]==0 for t in r["trajectories"])]
            universals=[r["seed"] for r in rs if r["universal_state_certificate"]["status"]=="all_initial_states_visible_target_certified"]
            phase=[r["seed"] for r in rs if r["observed_phase_dependence"]]
            exact=set(computed_conditions[c]["success_seeds_2000"]) if k==2000 else set()
            runtime_groups[c][k]={"all_66_exact_at_20":len(fully),"all_66_exact_seeds":fully,"universal_status_counts":dict(cert),
                 "universal_eventual_success_seeds":universals,"observed_phase_dependence_seeds":phase,
                 "primary_exact_without_eventual_universal_certificate":sorted(exact-set(universals)) if k==2000 else [],
                 "eventual_universal_onsets":{r["seed"]:r["universal_state_certificate"].get("certified_visible_settling_tick") for r in rs if r["seed"] in universals}}
    certs=Counter(r["universal_state_certificate"]["status"] for r in runtime)
    trajectories=[t for r in runtime for t in r["trajectories"]]
    assert len(trajectories)==33792
    recomputed={"labeled_modes":512,"labeled_trajectories":33792,
        "fixed_readout_all_66_exact_modes":sum(all(t["errors_at_20"]==0 for t in r["trajectories"]) for r in runtime),
        "observed_phase_dependent_modes":sum(r["observed_phase_dependence"] for r in runtime),
        "initialization_counterexample_modes":sum(r["initialization_counterexample"] is not None for r in runtime),
        "universal_correct_modes":certs["all_initial_states_visible_target_certified"],
        "universal_persistent_failure_modes":certs["persistent_visible_failure_certified"],
        "universal_inconclusive_modes":certs["inconclusive"],
        "orbit_certified_correct_trajectories":sum(t["orbit_certified_correct_onset"] is not None for t in trajectories),
        "mixed_phase_trajectories":sum(t["cycle_behavior"]=="some_phases_correct" for t in trajectories),
        "persistent_failure_trajectories":sum(t["cycle_behavior"]=="no_phases_correct" for t in trajectories),
        "unresolved_trajectories":sum(t["cycle_period"] is None for t in trajectories),
        "unique_rule_profiles":len({r["source"]["rule_sha256"] for r in runtime})}
    assert recomputed == result["runtime_summary"],(recomputed,result["runtime_summary"])
    # Check the original-probe runtime endpoints against independent run records.
    for r in runtime:
        m=manifests[r["seed"],r["condition"]]
        e=next(e for e in m["evaluations"] if e["global_update"]==r["global_update"])
        errors=sum(t["errors_at_20"] for t in r["trajectories"] if t["group"]=="original_probe")
        assert errors==e[r["hardening"]]["bit_errors"]
    preflight=read(bundle / "results/R004_preflight_artifact_final/validation.json")
    assert preflight["status"]=="passed"
    for c,case in preflight["conditions"].items():
        assert all(case["historical_replay_450_to_500"].values())
        assert all(case["interruption_parity"].values())
        for step in case["budget_field_parity"]:
            assert all(v for k,v in step.items() if k!="global_update")
    for rel,rec in preflight["code"]["implementation_sources"].items():
        verify(bundle / "implementation" / rel,rec)
    return {"scope":"Independent record/hash/statistical review; training and GPU preflight not rerun. Boolean subset replay is reported separately.",
       "inventory_files_verified":len(inventory["files"]),"source_run_count":64,"parent_manifests_verified":64,
       "per_update_records":count_updates,"new_checkpoint_references":count_checkpoints,"new_export_references":count_exports,
       "batch_stream_pairs_verified":16,"matching_training_sources":dict(matching_train_sources),"training_record_analysis_source_versions":dict(analysis_hash_versions),
       "preflight_record_checks":"passed; source-matching final preflight read, not re-executed",
       "original_preflight_record_in_bundle":(bundle / formal["preflight"]).exists(),
       "synchronized_update_seconds":sum(x["synchronized_seconds"] for x in metric_stats.values()),
       "summed_training_wall_seconds":sum(x["wall_seconds"] for x in metric_stats.values()),
       "condition_summary":computed_conditions,"hardening_gap":hardening_gap,"paired":paired,"runtime_summary":recomputed,"runtime_by_condition_update":runtime_groups,
       "per_seed":detail,"source_identities":identities}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=audit(args.bundle,args.reference)
    (args.output/"review_metrics.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"validation":"passed","conditions":{c:{"success_500":r["common_exact_500"],"success_2000":r["common_exact_2000"],"size_500":r["common_sizes"][500],"size_2000":r["common_sizes"][2000],"any_relapse":r["any_saved_history_relapse_seeds"],"soft_improved":r["soft_loss_improved_count"],"step_seconds_median":r["median_synchronized_seconds_per_update"]} for c,r in result["condition_summary"].items()},"paired":result["paired"],"runtime_2000":{c:r[2000] for c,r in result["runtime_by_condition_update"].items()}},indent=2))


if __name__=="__main__":
    main()
