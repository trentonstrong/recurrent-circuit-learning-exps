#!/usr/bin/env python3
"""Independently check the committed runtime records and derive cohort summaries."""
from __future__ import annotations
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULT = ROOT / "results/R002_formal_attempt01_runtime_profile_attempt01"
CONDITIONS = ["categorical_reference_decay", "truth_reference_decay",
              "categorical_no_decay", "truth_no_decay"]
GROUPS = {"original_probe": 32, "fresh_probe": 32, "all_zero": 1, "all_one": 1}
CORRECT = "all_initial_states_visible_target_certified"
WRONG = "persistent_visible_failure_certified"

def read(name):
    return json.loads((RESULT / name).read_text())

def phase_correct(row, tick):
    mu, period = row["cycle_entry_tick"], row["cycle_period"]
    assert mu is not None and tick >= mu
    return (tick - mu) % period in row["correct_phase_indices"]

def main():
    manifest = read("manifest.json")
    for item in manifest["result_files"]:
        data = (ROOT / item["path"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    config_bytes = (ROOT / manifest["config_path"]).read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == manifest["config_sha256"]
    assert json.loads(config_bytes) == manifest["config"]
    fresh = np.random.Generator(np.random.PCG64(20260910)).integers(
        0, 2, size=(32, 16, 16, 8), dtype=np.uint8)
    assert hashlib.sha256(fresh.tobytes()).hexdigest() == manifest["fresh_probe"]["raw_array_sha256"]

    modes = read("runs.json")["modes"]
    published = read("aggregate.json")
    with gzip.open(RESULT / "trajectories.jsonl.gz", "rt") as handle:
        trajectories = [json.loads(line) for line in handle]
    by_mode = collections.defaultdict(list)
    for row in trajectories:
        by_mode[row["run_id"], row["hardening"]].append(row)
    assert len(modes) == 128 and len(trajectories) == 8448
    assert len(by_mode) == 128
    assert {(m["seed"], m["condition"], m["hardening"]) for m in modes} == {
        (seed, condition, hard) for seed in range(16)
        for condition in CONDITIONS for hard in ("common", "native")}
    assert len({m["source"]["rule_sha256"] for m in modes}) == 96
    assert sum(m["profile_reused"] for m in modes) == 32
    verified_rows = 0
    for mode in modes:
        rows = by_mode[mode["run_id"], mode["hardening"]]
        assert len(rows) == 66 and len({r["initialization_id"] for r in rows}) == 66
        for row in rows:
            assert row["condition"] == mode["condition"]
            assert 0 <= row["errors_at_20"] <= 256
            assert row["certified_visible_settling_tick"] == mode["universal_state_certificate"]["certified_visible_settling_tick"]
            hit, suffix = row["first_hit_tick"], row["observed_correct_suffix_start"]
            assert hit is None or 0 <= hit <= 256
            assert suffix is None or hit is not None and hit <= suffix <= 256
            assert row["observed_correct_suffix_length"] == (0 if suffix is None else 257 - suffix)
            mu, period = row["cycle_entry_tick"], row["cycle_period"]
            if mu is None:
                assert period is None and row["cycle_behavior"] == "unresolved_by_cap"
                assert not row["evidence"]["exact_full_state_recurrence"]
            else:
                assert 0 <= mu < mu + period <= 256
                assert period % row["visible_cycle_period"] == 0
                phases = row["correct_phase_indices"]
                assert phases == sorted(set(phases))
                assert all(0 <= p < period for p in phases)
                assert row["cycle_target_fraction"] == len(phases) / period
                expected = "all_phases_correct" if len(phases) == period else "some_phases_correct" if phases else "no_phases_correct"
                assert row["cycle_behavior"] == expected
                assert 0 <= row["cycle_min_errors"] <= row["cycle_mean_errors"] <= row["cycle_max_errors"] <= 256
                assert row["phase_at_tick_20"] == ((20 - mu) % period if 20 >= mu else None)
                if mu <= 20:
                    assert phase_correct(row, 20) == (row["errors_at_20"] == 0)
                assert phase_correct(row, 256) == (suffix is not None)
                if expected == "all_phases_correct":
                    assert suffix is not None and suffix <= mu
                    assert row["cycle_max_errors"] == 0
                elif expected == "some_phases_correct":
                    k = 256
                    while k >= mu and phase_correct(row, k):
                        k -= 1
                    assert row["observed_correct_suffix_length"] == 256 - k
                else:
                    assert row["cycle_min_errors"] > 0 and suffix is None
                if hit is not None and hit >= mu:
                    assert hit == mu + min(phases)
            if mode["universal_state_certificate"]["status"] == CORRECT:
                assert row["cycle_behavior"] != "some_phases_correct"
                assert row["cycle_behavior"] != "no_phases_correct"
                assert suffix is not None and suffix <= row["certified_visible_settling_tick"]
            if mode["universal_state_certificate"]["status"] == WRONG:
                assert row["cycle_behavior"] in ("no_phases_correct", "unresolved_by_cap")
            verified_rows += 1
        for group, count in GROUPS.items():
            selected = [r for r in rows if r["group"] == group]
            assert len(selected) == count
            summary = mode["initialization_groups"][group]
            assert sum(r["errors_at_20"] for r in selected) == summary["report_ticks"]["20"]["aggregate_errors"]
            assert sum(r["errors_at_20"] == 0 for r in selected) == summary["report_ticks"]["20"]["perfect_initializations"]
            signatures = {r["cycle_signature_sha256"] for r in selected if r["cycle_signature_sha256"] is not None}
            assert len(signatures) == summary["known_distinct_full_cycles"]
            assert sum(r["cycle_period"] is None for r in selected) == summary["unresolved_trajectory_count"]
            suffixes = [r["observed_correct_suffix_start"] for r in selected]
            assert summary["all_have_observed_correct_suffix"] == all(s is not None for s in suffixes)
            assert summary["latest_observed_correct_suffix_start"] == (max(suffixes) if all(s is not None for s in suffixes) else None)
            for tick, report in summary["report_ticks"].items():
                assert report["all_correct"] == (report["aggregate_errors"] == 0)
                assert report["all_correct"] == (report["perfect_initializations"] == count)
                if all(r["cycle_entry_tick"] is not None and int(tick) >= r["cycle_entry_tick"] for r in selected):
                    assert sum(phase_correct(r, int(tick)) for r in selected) == report["perfect_initializations"]

    primary = [m for m in modes if m["hardening"] == "common"]
    # Recompute every published table row, including the secondary hardening.
    for published_row in published["main_table"]:
        selected = [m for m in modes if m["condition"] == published_row["condition"] and m["hardening"] == published_row["hardening"]]
        assert len(selected) == published_row["run_count"] == 16
        assert sum(m["universal_state_certificate"]["status"] == CORRECT for m in selected) == published_row["universally_eventually_correct_count"]
        assert sum(m["universal_state_certificate"]["status"] == WRONG for m in selected) == published_row["certified_persistent_failure_count"]
        for tick, count in published_row["original_all_correct_run_count_by_tick"].items():
            assert sum(m["initialization_groups"]["original_probe"]["report_ticks"][tick]["all_correct"] for m in selected) == count
        assert sum(any(r["cycle_behavior"] == "some_phases_correct" for r in by_mode[m["run_id"],m["hardening"]]) for m in selected) == published_row["mixed_cycle_run_count"]
        assert sum(any(r["cycle_period"] is None for r in by_mode[m["run_id"],m["hardening"]]) for m in selected) == published_row["run_with_unresolved_trajectory_count"]
    with (RESULT / "per_seed.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 128
    for row in csv_rows:
        mode = next(m for m in modes if m["seed"] == int(row["seed"]) and m["condition"] == row["condition"] and m["hardening"] == row["hardening"])
        assert int(row["visible_core_binary_nodes"]) == mode["structure"]["visible_core_binary_nodes"]
        for group, field in [("original_probe","original_all_correct_ticks"),("fresh_probe","fresh_all_correct_ticks"),("all_zero","all_zero_correct_ticks"),("all_one","all_one_correct_ticks")]:
            actual = {int(t) for t in row[field].split(";") if t}
            expected = {int(t) for t,v in mode["initialization_groups"][group]["report_ticks"].items() if v["all_correct"]}
            assert actual == expected
    old = json.loads((ROOT / "results/R002_formal_attempt01/analysis.json").read_text())
    for condition in CONDITIONS:
        success_seeds = sorted(m["seed"] for m in primary if m["condition"] == condition and m["initialization_groups"]["original_probe"]["report_ticks"]["20"]["all_correct"])
        assert success_seeds == old["conditions"][condition]["success_seeds"]
    detailed = []
    for mode in primary:
        rows = by_mode[mode["run_id"], "common"]
        status = mode["universal_state_certificate"]["status"]
        behaviors = collections.Counter(r["cycle_behavior"] for r in rows)
        if status == CORRECT:
            classification = "universal_eventual_target"
        elif status == WRONG:
            classification = "universal_persistent_wrong_bit"
        elif behaviors["some_phases_correct"]:
            classification = "mixed_correct_wrong_cycle_phases"
        elif behaviors["all_phases_correct"] == 66:
            classification = "all_66_orbits_eventually_correct"
        elif behaviors["unresolved_by_cap"]:
            classification = "unresolved_orbits_and_universal_test"
        elif behaviors["no_phases_correct"] == 66:
            classification = "all_66_orbits_eventually_wrong"
        else:
            classification = "different_correct_wrong_basins"
        g = mode["initialization_groups"]
        detailed.append({
            "run_id": mode["run_id"], "seed": mode["seed"], "condition": mode["condition"],
            "classification": classification, "cycle_behavior_counts": dict(behaviors),
            "original_success_at_20": g["original_probe"]["report_ticks"]["20"]["all_correct"],
            "original_success_at_64": g["original_probe"]["report_ticks"]["64"]["all_correct"],
            "all_66_success_at_64": all(v["report_ticks"]["64"]["all_correct"] for v in g.values()),
            "certified_settling_tick": mode["universal_state_certificate"]["certified_visible_settling_tick"],
            "first_simultaneous_hit_by_group": {k: v["first_simultaneous_hit_tick"] for k, v in g.items()},
            "visible_core_binary_nodes": mode["structure"]["visible_core_binary_nodes"],
            "visible_core_state_channels": mode["structure"]["visible_recurrent_channels"],
        })
    table = []
    for condition in CONDITIONS:
        ms = [m for m in primary if m["condition"] == condition]
        entries = [m for m in detailed if m["condition"] == condition]
        sizes = [m["visible_core_binary_nodes"] for m in entries]
        table.append({"condition": condition,
            "tick20_original_successes": sum(m["original_success_at_20"] for m in entries),
            "tick64_original_successes": sum(m["original_success_at_64"] for m in entries),
            "tick64_all66_successes": sum(m["all_66_success_at_64"] for m in entries),
            "universal_correct": sum(m["classification"] == "universal_eventual_target" for m in entries),
            "all_original_orbits_certified_eventually_correct": sum(all(r["cycle_behavior"] == "all_phases_correct" for r in by_mode[m["run_id"],"common"] if r["group"] == "original_probe") for m in ms),
            "core_size_median": float(np.median(sizes)), "core_size_min": min(sizes), "core_size_max": max(sizes),
            "disjoint_classification_counts": dict(collections.Counter(m["classification"] for m in entries))})
    for row in table:
        old = next(r for r in published["main_table"] if r["condition"] == row["condition"] and r["hardening"] == "common")
        assert row["tick20_original_successes"] == old["original_tick20_success_count"]
        assert row["tick64_original_successes"] == old["original_all_correct_run_count_by_tick"]["64"]
        assert row["universal_correct"] == old["universally_eventually_correct_count"]
    structure = {}
    for decay in ("reference_decay", "no_decay"):
        cats = {m["seed"]: m for m in detailed if m["condition"] == "categorical_" + decay}
        truth = {m["seed"]: m for m in detailed if m["condition"] == "truth_" + decay}
        diffs = [truth[s]["visible_core_binary_nodes"] - cats[s]["visible_core_binary_nodes"] for s in range(16)]
        structure[decay] = {"paired_truth_minus_categorical_core_sizes": diffs,
                            "truth_larger_pairs": sum(d > 0 for d in diffs)}
    phase = []
    for seed in (2, 14):
        run_id = f"R002_formal_attempt01_seed{seed:02d}_truth_reference_decay"
        rows = by_mode[run_id, "common"]
        phase.append({"seed": seed, "run_id": run_id,
            "full_periods": sorted({r["cycle_period"] for r in rows}),
            "visible_periods": sorted({r["visible_cycle_period"] for r in rows}),
            "target_fractions": sorted({r["cycle_target_fraction"] for r in rows}),
            "perfect_counts_at_ticks_64_through_79": {group: [sum(phase_correct(r,t) for r in rows if r["group"]==group) for t in range(64,80)] for group in GROUPS}})
    report = {"source_commit": "60ef04441ff16e26e395e28b9c5c63cb7d2849af",
        "verification": {"result_files_hash_verified": 9, "trajectory_records_checked": verified_rows,
            "mode_count": len(modes), "unique_rule_hashes": 96, "fresh_probe_raw_bytes_reconstructed": True,
            "scope": "Committed records and their internal mathematical consistency; separate subset replay verifies available Boolean source payloads."},
        "primary_common_table": table,
        "primary_disjoint_counts": dict(collections.Counter(m["classification"] for m in detailed)),
        "structure_pairs": structure, "phase_examples": phase, "primary_modes": detailed}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("primary_modes",)}, indent=2))

if __name__ == "__main__":
    main()
