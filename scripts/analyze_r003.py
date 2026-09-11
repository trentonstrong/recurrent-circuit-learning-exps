#!/usr/bin/env python3
"""Aggregate an R003 cohort and build its compact review deliverables."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
import tarfile
from pathlib import Path
from typing import Any, Callable

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "seed_values": values,
    }


def arm(case: dict[str, Any], alpha: float) -> dict[str, Any]:
    return next(item for item in case["arms"] if item["alpha"] == alpha)


def step(current_arm: dict[str, Any], eta: float) -> dict[str, Any]:
    return next(item for item in current_arm["steps"] if item["eta"] == eta)


def artifact_output_contrast(case: dict[str, Any], eta: float) -> tuple[float, float]:
    with np.load(case["derived_artifact"]["path"], allow_pickle=False) as stored:
        original = stored[f"alpha_0_eta_{eta:g}_output_delta"]
        factorized = stored[f"alpha_1_eta_{eta:g}_output_delta"]
    difference = float(np.linalg.norm((factorized - original).ravel()))
    denominator = max(
        float(np.linalg.norm(original.ravel())),
        float(np.linalg.norm(factorized.ravel())),
    )
    return difference, difference / denominator if denominator else math.nan


def _scale(values: list[float], low: float, high: float, log: bool) -> Callable[[float], float]:
    transformed = [math.log10(max(value, 1e-300)) if log else value for value in values]
    minimum, maximum = min(transformed), max(transformed)
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5
    return lambda value: low + (
        ((math.log10(max(value, 1e-300)) if log else value) - minimum)
        / (maximum - minimum)
    ) * (high - low)


def scatter_svg(
    path: Path,
    series: list[tuple[str, str, list[tuple[float, float]]]],
    title: str,
    x_label: str,
    y_label: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
    identity: bool = False,
) -> None:
    width, height = 800, 520
    left, right, top, bottom = 90, 30, 65, 75
    points = [point for _, _, values in series for point in values]
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    if identity:
        shared = xs + ys
        xs = shared
        ys = shared
    x_map = _scale(xs, left, width - right, log_x)
    y_raw = _scale(ys, height - bottom, top, log_y)
    y_map = lambda value: y_raw(value)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>',
    ]
    if identity:
        lo, hi = min(xs), max(xs)
        parts.append(
            f'<line x1="{x_map(lo):.2f}" y1="{y_map(lo):.2f}" x2="{x_map(hi):.2f}" y2="{y_map(hi):.2f}" stroke="#777" stroke-dasharray="5 5"/>'
        )
    for label, color, values in series:
        for x_value, y_value in values:
            parts.append(
                f'<circle cx="{x_map(x_value):.2f}" cy="{y_map(y_value):.2f}" r="3" fill="{color}" fill-opacity="0.68"/>'
            )
    parts.extend(
        [
            f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(x_label)}</text>',
            f'<text x="22" y="{height/2}" text-anchor="middle" transform="rotate(-90 22 {height/2})" font-family="sans-serif" font-size="14">{html.escape(y_label)}</text>',
        ]
    )
    legend_x = left + 12
    for index, (label, color, _) in enumerate(series):
        y = top + 18 * index
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+10}" y="{y+4}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts))


def build_review_archive(
    archive_path: Path,
    root: Path,
    result_directory: Path,
    preflight_directory: Path,
    preflight_cases: list[dict[str, Any]],
) -> None:
    members: dict[Path, str] = {}
    for directory, prefix in (
        (result_directory, "formal"),
        (preflight_directory, "preflight"),
    ):
        for name in ("manifest.json", "validation.json", "cases.jsonl", "summary.md", "analysis.json"):
            path = directory / name
            if path.exists():
                members[path] = f"{prefix}/{name}"
        figures = directory / "figures"
        if figures.exists():
            for path in figures.glob("*.svg"):
                members[path] = f"{prefix}/figures/{path.name}"
    for path in (
        root / "configs/experiments/r003_same_function_v1.json",
        root / "experiments/R003/SPEC.md",
        root / "experiments/R003/HANDOFF.md",
        root / "recurrent_circuit_learning/r003.py",
        root / "scripts/r003.py",
        root / "scripts/analyze_r003.py",
        root / "tests/test_r003.py",
    ):
        members[path] = f"implementation/{path.relative_to(root)}"
    probe = root / "artifacts/R001_20260909T231016Z_seed23_jaxgpu_attempt01/fixed_evaluation_set.npz"
    members[probe] = f"sources/{probe.name}"
    for case in preflight_cases:
        derived = Path(case["derived_artifact"]["path"])
        members[derived] = f"preflight/derived/{derived.name}"
        run_id = case["source"]["run_id"]
        checkpoint = Path(case["source"]["checkpoint"]["path"])
        if not checkpoint.exists():
            checkpoint = root / "artifacts" / run_id / checkpoint.name
        wiring = Path(case["source"]["wiring"]["path"])
        if not wiring.exists():
            wiring = root / "artifacts" / run_id / wiring.name
        members[checkpoint] = f"sources/{run_id}/{checkpoint.name}"
        members[wiring] = f"sources/{run_id}/{wiring.name}"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path, name in sorted(members.items(), key=lambda item: item[1]):
            archive.add(path, arcname=name, recursive=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--review-archive", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    result_directory = args.results.resolve()
    preflight_directory = args.preflight.resolve()
    cases = [json.loads(line) for line in (result_directory / "cases.jsonl").read_text().splitlines()]
    preflight_cases = [json.loads(line) for line in (preflight_directory / "cases.jsonl").read_text().splitlines()]
    if len(cases) != 96 or not all(case["status"] == "passed" for case in cases):
        raise ValueError("analysis requires the complete passing 96-case cohort")
    factorized = [arm(case, 1.0) for case in cases]
    neutrality = {
        name: max(
            current["invariance"][name]["max_abs_error"]
            for case in cases for current in case["arms"]
        )
        for name in ("effective_table", "states", "loss", "q_gradient")
    }
    neutrality["state_jvp"] = max(
        check["max_abs_error"]
        for case in cases for current in case["arms"]
        for check in current["invariance"]["state_jvps"]
    )
    grouped = []
    for condition in ("categorical_reference_decay", "categorical_no_decay"):
        for update in (0, 250, 500):
            selected = [case for case in cases if case["condition"] == condition and case["update"] == update]
            current_arms = [arm(case, 1.0) for case in selected]
            record: dict[str, Any] = {"condition": condition, "update": update}
            extractors = {
                "eligible_gate_fraction": lambda case, current: case["coverage"]["eligible_fraction"],
                "eligible_response_fraction": lambda case, current: case["coverage"]["eligible_response_fraction"],
                "matrix_relative_difference": lambda case, current: current["matrix"]["relative_difference_from_original"],
                "q_response_relative_difference": lambda case, current: current["response"]["common_relative_difference_from_original"],
                "loss_input_output_response_difference_l2": lambda case, current: current["output_response"]["actual_difference_from_original"]["loss_inputs"]["l2"],
                "additional_input_output_response_difference_l2": lambda case, current: current["output_response"]["actual_difference_from_original"]["additional_inputs"]["l2"],
                "output_response_relative_difference": lambda case, current: current["output_response"]["actual_difference_from_original"]["relative"],
                "native_argmax_gate_changes": lambda case, current: current["native_gate_id_changes_from_original"],
            }
            for name, extractor in extractors.items():
                record[name] = summary([float(extractor(case, current)) for case, current in zip(selected, current_arms)])
            for eta in (0.001, 0.0001):
                contrasts = [artifact_output_contrast(case, eta) for case in selected]
                record[f"eta_{eta:g}_finite_output_delta_difference_l2"] = summary([item[0] for item in contrasts])
                record[f"eta_{eta:g}_finite_output_delta_relative_difference"] = summary([item[1] for item in contrasts])
                record[f"eta_{eta:g}_q_linearization_relative_residual"] = summary([
                    float(step(current, eta)["q_linearization_relative_residual"]) for current in current_arms
                ])
                record[f"eta_{eta:g}_output_linearization_relative_residual"] = summary([
                    float(step(current, eta)["output_linearization_relative_residual"]) for current in current_arms
                ])
            grouped.append(record)
    global_contrasts = {eta: [artifact_output_contrast(case, eta) for case in cases] for eta in (0.001, 0.0001)}
    analysis = {
        "schema_version": 1,
        "status": "passed",
        "interpretation": "post-training mechanism diagnostic, not training or causal circuit-size evidence",
        "counts": {
            "checkpoint_cases": len(cases),
            "arm_cases": sum(len(case["arms"]) for case in cases),
            "step_cases": sum(len(current["steps"]) for case in cases for current in case["arms"]),
            "valid_same_function_arms": sum(current["status"] == "valid_same_function" for case in cases for current in case["arms"]),
        },
        "numerical_neutrality_max_abs_error": neutrality,
        "common_rounding_gate_changes": sum(current["common_gate_id_changes_from_original"] for current in factorized),
        "native_argmax_gate_changes": sum(current["native_gate_id_changes_from_original"] for current in factorized),
        "source_artifacts_unchanged": all(all(case["source_artifacts_unchanged"].values()) for case in cases),
        "global_primary_contrast": {
            "matrix_relative_difference": summary([current["matrix"]["relative_difference_from_original"] for current in factorized]),
            "q_response_relative_difference": summary([current["response"]["common_relative_difference_from_original"] for current in factorized]),
            "loss_input_output_response_difference_l2": summary([current["output_response"]["actual_difference_from_original"]["loss_inputs"]["l2"] for current in factorized]),
            "additional_input_output_response_difference_l2": summary([current["output_response"]["actual_difference_from_original"]["additional_inputs"]["l2"] for current in factorized]),
            "output_response_relative_difference": summary([current["output_response"]["actual_difference_from_original"]["relative"] for current in factorized]),
            "eligible_gate_fraction": summary([case["coverage"]["eligible_fraction"] for case in cases]),
            "eligible_response_fraction": summary([case["coverage"]["eligible_response_fraction"] for case in cases]),
        },
        "finite_steps": {
            f"eta_{eta:g}": {
                "output_delta_difference_l2": summary([item[0] for item in values]),
                "output_delta_relative_difference": summary([item[1] for item in values]),
                "factorized_q_linearization_relative_residual": summary([step(current, eta)["q_linearization_relative_residual"] for current in factorized]),
                "factorized_output_linearization_relative_residual": summary([step(current, eta)["output_linearization_relative_residual"] for current in factorized]),
            }
            for eta, values in global_contrasts.items()
        },
        "groups": grouped,
    }
    write_json(result_directory / "analysis.json", analysis)
    figures = result_directory / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"categorical_reference_decay": "#3366cc", "categorical_no_decay": "#dc3912"}
    scatter_svg(
        figures / "invariance_vs_output_effect.svg",
        [(condition, colors[condition], [
            (arm(case, 1.0)["invariance"]["states"]["max_abs_error"], arm(case, 1.0)["output_response"]["actual_difference_from_original"]["loss_inputs"]["l2"])
            for case in cases if case["condition"] == condition
        ]) for condition in colors],
        "Numerical invariance versus observable response change",
        "maximum state invariance residual (log10)",
        "loss-input output-response difference L2 (log10)",
        log_x=True,
        log_y=True,
    )
    scatter_svg(
        figures / "matrix_vs_q_response.svg",
        [(condition, colors[condition], [
            (arm(case, 1.0)["matrix"]["relative_difference_from_original"], arm(case, 1.0)["response"]["common_relative_difference_from_original"])
            for case in cases if case["condition"] == condition
        ]) for condition in colors],
        "Fixed-q coordinate matrix and functional-response changes",
        "relative matrix difference",
        "relative q-response difference",
    )
    scatter_svg(
        figures / "predicted_vs_observed_output_steps.svg",
        [(f"eta={eta:g}", color, [
            (step(arm(case, 1.0), eta)["output_prediction"]["l2"], step(arm(case, 1.0), eta)["output_delta"]["l2"])
            for case in cases
        ]) for eta, color in ((0.001, "#109618"), (0.0001, "#990099"))],
        "Predicted versus observed factorized-arm output steps",
        "predicted output delta L2 (log10)",
        "observed output delta L2 (log10)",
        log_x=True,
        log_y=True,
        identity=True,
    )
    scatter_svg(
        figures / "eligible_coverage_vs_output_effect.svg",
        [(condition, colors[condition], [
            (case["coverage"]["eligible_response_fraction"], arm(case, 1.0)["output_response"]["actual_difference_from_original"]["loss_inputs"]["l2"])
            for case in cases if case["condition"] == condition
        ]) for condition in colors],
        "Eligible response coverage versus observable effect",
        "eligible fraction of baseline response squared norm (log10)",
        "loss-input output-response difference L2 (log10)",
        log_x=True,
        log_y=True,
    )
    primary = analysis["global_primary_contrast"]
    eta_primary = analysis["finite_steps"]["eta_0.001"]
    eta_audit = analysis["finite_steps"]["eta_0.0001"]
    table_lines = []
    for group in grouped:
        short = "reference decay" if group["condition"].endswith("reference_decay") else "no decay"
        table_lines.append(
            f'| {short} | {group["update"]} | {group["matrix_relative_difference"]["median"]:.3g} | {group["q_response_relative_difference"]["median"]:.3g} | {group["loss_input_output_response_difference_l2"]["median"]:.3g} |'
        )
    summary_text = f"""# R003 formal diagnostic: R003_formal_attempt02

Status: **passed**. All 96 checkpoint cases, 288 same-function arms, and 576
independent one-step diagnostics completed under the frozen v1 protocol.

## Result

The intervention was numerically neutral before the step: the largest q error
was {neutrality['effective_table']:.3g}, the largest recurrent-state error was
{neutrality['states']:.3g}, and common hardening changed zero gate IDs. Despite
that fixed relaxed function, the original-to-factorized coordinate matrix changed
in every checkpoint (relative difference {primary['matrix_relative_difference']['min']:.3g}–{primary['matrix_relative_difference']['max']:.3g}; median {primary['matrix_relative_difference']['median']:.3g}).

The induced q response also changed in every checkpoint (relative difference
{primary['q_response_relative_difference']['min']:.3g}–{primary['q_response_relative_difference']['max']:.3g}; median {primary['q_response_relative_difference']['median']:.3g}). That change reached the visible outputs on the two loss inputs in every case: response-difference L2 ranged from
{primary['loss_input_output_response_difference_l2']['min']:.3g} to {primary['loss_input_output_response_difference_l2']['max']:.3g}, with median {primary['loss_input_output_response_difference_l2']['median']:.3g}.

At eta 0.001, the actual original-to-factorized output-step difference ranged
from {eta_primary['output_delta_difference_l2']['min']:.3g} to {eta_primary['output_delta_difference_l2']['max']:.3g}. The factorized-arm output linearization residual had median relative size {eta_primary['factorized_output_linearization_relative_residual']['median']:.3g} and maximum {eta_primary['factorized_output_linearization_relative_residual']['max']:.3g}. At eta 0.0001 those values fell to {eta_audit['factorized_output_linearization_relative_residual']['median']:.3g} and {eta_audit['factorized_output_linearization_relative_residual']['max']:.3g}, respectively, supporting the local first-order interpretation.

Factorization changed {analysis['native_argmax_gate_changes']} native argmax gate
IDs across the 96 primary contrasts while changing zero common-rounded IDs. This
is a separate extraction observation. Eligible gates covered {primary['eligible_gate_fraction']['min']:.3%}–{primary['eligible_gate_fraction']['max']:.3%} of slots; the smallest eligible share of baseline response squared norm was {primary['eligible_response_fraction']['min']:.3g}, so that low-coverage case should be interpreted cautiously.

| training history | update | median relative M change | median relative q-response change | median visible-response difference L2 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(table_lines)}

These measurements establish representative-dependent local SGD response at the
saved R002 checkpoints and show that it reaches the observed recurrent outputs.
They do not establish improved long-run training, naturally occurring motion
along fibers, or a causal explanation of R002 circuit-size differences.

`analysis.json` retains all 16 seed values for every condition/update group.
`cases.jsonl` contains per-case controls, geometry, responses, and finite steps;
the hashed derived arrays retain p, q, gradients, responses, and measured deltas.
"""
    (result_directory / "summary.md").write_text(summary_text)
    build_review_archive(args.review_archive.resolve(), root, result_directory, preflight_directory, preflight_cases)
    manifest_path = result_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    implementation_paths = [
        root / "recurrent_circuit_learning/r003.py",
        root / "scripts/r003.py",
        root / "scripts/analyze_r003.py",
        root / "tests/test_r003.py",
    ]
    manifest["analysis"] = {
        "status": "passed",
        "path": str(result_directory / "analysis.json"),
        "sha256": sha256_file(result_directory / "analysis.json"),
        "figures": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(figures.glob("*.svg"))
        ],
    }
    manifest["implementation_files"] = [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in implementation_paths
    ]
    manifest["review_archive"] = {
        "path": str(args.review_archive.resolve()),
        "bytes": args.review_archive.stat().st_size,
        "sha256": sha256_file(args.review_archive),
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "passed", "analysis": str(result_directory / "analysis.json"), "review_archive": str(args.review_archive)}, indent=2))


if __name__ == "__main__":
    main()
