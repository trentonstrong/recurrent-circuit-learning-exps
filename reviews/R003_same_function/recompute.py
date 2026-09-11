#!/usr/bin/env python3
"""Audit committed R003 records and derive response angles without JAX or payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

CONDITIONS = ("categorical_reference_decay", "categorical_no_decay")
SOURCE_COMMIT = "fe14ca65cf62cdc3e283cf8891a745a46967bd42"
CASE_BLOB = "035a4f59562b2e743c2f4050bd8de3ae5a7af264"


def stats(values):
    return dict(count=len(values), min=min(values), median=statistics.median(values), max=max(values))


def distribution(values):
    return {**stats(values), "seed_values": values}


def get_arm(case, alpha):
    return next(a for a in case["arms"] if a["alpha"] == alpha)


def get_step(arm, eta):
    return next(s for s in arm["steps"] if s["eta"] == eta)


def angle_and_ratio(a, b, difference):
    if not a or not b:
        return {"angle_degrees": None, "norm_ratio": None, "reason": "zero_norm"}
    cosine = (a*a + b*b - difference*difference) / (2*a*b)
    if abs(cosine) > 1 + 1e-10:
        raise ValueError("Inconsistent recorded norms")
    return {
        "angle_degrees": math.degrees(math.acos(max(-1., min(1., cosine)))),
        "norm_ratio": b/a,
        "reason": None,
    }


def output_norm(arm, subset):
    out = arm["output_response"]
    return out["common_" + subset]["l2"] if subset != "all" else math.hypot(
        out["common_loss_inputs"]["l2"], out["common_additional_inputs"]["l2"])


def output_difference(arm, subset):
    out = arm["output_response"]["common_difference_from_original"]
    return out[subset]["l2"] if subset != "all" else math.hypot(
        out["loss_inputs"]["l2"], out["additional_inputs"]["l2"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = root / "results/R003_formal_attempt02"
    raw = (result / "cases.jsonl").read_bytes()
    assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == CASE_BLOB
    cases = [json.loads(line) for line in raw.splitlines()]
    analysis = json.loads((result / "analysis.json").read_text())
    manifest = json.loads((result / "manifest.json").read_text())
    records = list(manifest["implementation_files"]) + [
        {"path": manifest["config_path"], "sha256": manifest["config_sha256"]},
        {"path": "results/R003_formal_attempt02/analysis.json", "sha256": manifest["analysis"]["sha256"]},
    ]
    verified_files = []
    for record in records:
        payload = (root / record["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        if "bytes" in record:
            assert len(payload) == record["bytes"]
        verified_files.append({k: record[k] for k in ("path", "sha256")})

    expected = {(seed, cond, k) for seed in range(16) for cond in CONDITIONS for k in (0, 250, 500)}
    assert len(cases) == len(expected) == 96
    assert {(c["seed"], c["condition"], c["update"]) for c in cases} == expected
    ledger = {(r["run_id"], r["update"]): r for r in manifest["source_ledger"]["cases"]}
    derived = {r["path"]: r for r in manifest["case_artifacts"]}
    for c in cases:
        assert c["status"] == "passed"
        assert all(c["source_artifacts_unchanged"].values())
        assert c["derived_artifact"] == derived[c["derived_artifact"]["path"]]
        record = ledger[c["source"]["run_id"], c["update"]]
        assert c["source"]["checkpoint"] == record["checkpoint"]
        assert c["source"]["wiring"] == record["wiring"]
        assert [a["alpha"] for a in c["arms"]] == [0, .5, 1]
        for a in c["arms"]:
            assert a["status"] == "valid_same_function"
            inv = a["invariance"]
            assert all(inv[k]["ok"] for k in ("effective_table", "states", "loss", "q_gradient"))
            assert all(v["ok"] for v in inv["state_jvps"])
            assert inv["normalization_ok"] and inv["common_rounding_ok"]
            assert a["common_gate_id_changes_from_original"] == 0
            assert {s["eta"] for s in a["steps"]} == {.001, .0001}

    extractors = {
        "eligible_gate_fraction": lambda c, a: c["coverage"]["eligible_fraction"],
        "eligible_response_fraction": lambda c, a: c["coverage"]["eligible_response_fraction"],
        "matrix_relative_difference": lambda c, a: a["matrix"]["relative_difference_from_original"],
        "q_response_relative_difference": lambda c, a: a["response"]["common_relative_difference_from_original"],
        "loss_input_output_response_difference_l2": lambda c, a: a["output_response"]["actual_difference_from_original"]["loss_inputs"]["l2"],
        "additional_input_output_response_difference_l2": lambda c, a: a["output_response"]["actual_difference_from_original"]["additional_inputs"]["l2"],
        "output_response_relative_difference": lambda c, a: a["output_response"]["actual_difference_from_original"]["relative"],
        "native_argmax_gate_changes": lambda c, a: a["native_gate_id_changes_from_original"],
    }
    for group in analysis["groups"]:
        selected = [c for c in cases if (c["condition"], c["update"]) == (group["condition"], group["update"])]
        for key, fn in extractors.items():
            assert distribution([fn(c, get_arm(c, 1)) for c in selected]) == group[key]
    for key, fn in extractors.items():
        if key in analysis["global_primary_contrast"]:
            assert distribution([fn(c, get_arm(c, 1)) for c in cases]) == analysis["global_primary_contrast"][key]
    for eta in (.001, .0001):
        for space in ("q", "output"):
            key = "factorized_" + space + "_linearization_relative_residual"
            vals = [get_step(get_arm(c, 1), eta)[space + "_linearization_relative_residual"] for c in cases]
            assert distribution(vals) == analysis["finite_steps"][f"eta_{eta:g}"][key]
    native = sum(get_arm(c, 1)["native_gate_id_changes_from_original"] for c in cases)
    assert native == analysis["native_argmax_gate_changes"] == 935

    preflight = [json.loads(line) for line in (root / "results/R003_preflight_attempt03/cases.jsonl").read_text().splitlines()]
    assert len(preflight) == 6
    assert all(a["full_logit_chain_rule"]["ok"] for c in preflight for a in c["arms"])

    rows = []
    for c in cases:
        a, b = get_arm(c, 0), get_arm(c, 1)
        record = {key: c[key] for key in ("seed", "condition", "update")}
        q_metrics = angle_and_ratio(a["response"]["common"]["l2"], b["response"]["common"]["l2"],
                                   b["response"]["common_difference_from_original"]["l2"])
        record.update(q_angle=q_metrics["angle_degrees"], q_norm_ratio=q_metrics["norm_ratio"])
        for subset in ("all", "loss_inputs", "additional_inputs"):
            metrics = angle_and_ratio(output_norm(a, subset), output_norm(b, subset), output_difference(b, subset))
            record[f"output_angle_{subset}"] = metrics["angle_degrees"]
            record[f"output_norm_ratio_{subset}"] = metrics["norm_ratio"]
        record.update(
            output_relative=output_difference(b, "all") / max(output_norm(a, "all"), output_norm(b, "all")),
            descent_ratio=b["response"]["d_loss_d_eta"] / a["response"]["d_loss_d_eta"],
            eligible_response_fraction=c["coverage"]["eligible_response_fraction"],
            residual_reduction=get_step(b, .001)["output_linearization_relative_residual"] /
                               get_step(b, .0001)["output_linearization_relative_residual"],
        )
        rows.append(record)
    scalar_keys = [k for k in rows[0] if k not in ("seed", "condition", "update")]
    groups = []
    for cond in CONDITIONS:
        for update in (0, 250, 500):
            selected = [r for r in rows if (r["condition"], r["update"]) == (cond, update)]
            groups.append({
                "condition": cond, "update": update,
                **{k: stats([r[k] for r in selected]) for k in scalar_keys},
                "larger_predicted_descent_count": sum(r["descent_ratio"] > 1 for r in selected),
            })
    precision_flags = [
        {**{k: c[k] for k in ("seed", "condition", "update")}, **c["fp32_to_promoted_fp64_outputs"]}
        for c in cases if not c["fp32_to_promoted_fp64_outputs"]["ok"]
    ]
    supplemental = {
        "source_commit": SOURCE_COMMIT,
        "case_blob_sha": CASE_BLOB,
        "case_file_sha256": hashlib.sha256(raw).hexdigest(),
        "verified_files": verified_files,
        "verification_scope": "Committed source and record audit; no source-checkpoint or derived-array replay; no independent rerun of the reported 27 tests.",
        "finite_step_scope": "Single-arm residual summaries verified from cases. Cross-arm finite-delta arrays unavailable; their published summaries were not independently replayed.",
        "angle_method": "Common-gradient q responses and shared-Q output responses; cos=(a*a+b*b-d*d)/(2*a*b), using committed vector and difference norms.",
        "counts": {"checkpoints": len(cases), "arms": 288, "steps": 576, "preflight_chain_checks": 18, "native_reset_gate_changes": native},
        "groups": groups, "cases": rows,
        "precision_comparison_flags": precision_flags,
        "residual_reduction": stats([r["residual_reduction"] for r in rows]),
    }
    output = args.output or Path(__file__).with_name("review_metrics.json")
    output.write_text(json.dumps(supplemental, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "passed_record_audit", "checkpoints": len(cases),
        "precision_flags": len(precision_flags), "native_changes": native,
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
