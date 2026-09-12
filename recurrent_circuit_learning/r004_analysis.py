"""Offline structural, runtime, and paired endpoint analysis for R004."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts import r002_runtime_profile as profile

from .r000 import sha256_file
from .r002 import CONDITIONS
from .r002_runner import _artifact
from .r004_runner import (
    code_provenance,
    continuation_batch_hashes_sha256,
    validate_config_contract,
    write_json,
)


def _verify_record(path: Path, record: dict[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != record["bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise ValueError(f"R004 analysis artifact identity mismatch: {path}")


def enumerate_runs(
    repository_root: Path,
    artifact_root: Path,
    sweep_id: str,
    config_sha256: str,
) -> list[dict[str, Any]]:
    manifests = []
    for path in sorted(
        (repository_root / "results").glob(f"R004_{sweep_id}_seed*_*/manifest.json")
    ):
        manifest = json.loads(path.read_text())
        if (
            manifest.get("experiment_id") != "R004"
            or manifest.get("status") != "completed"
            or manifest.get("current_update") != 2000
            or manifest.get("config_sha256") != config_sha256
        ):
            raise ValueError(f"incomplete R004 source: {path}")
        manifest["_manifest_path"] = path
        manifest["_artifact_directory"] = artifact_root / manifest["run_id"]
        parent_record = manifest["lineage"]["parent_manifest"]
        parent_path = Path(parent_record["path"])
        _verify_record(parent_path, parent_record)
        parent_manifest = json.loads(parent_path.read_text())
        if (
            parent_manifest.get("run_id") != manifest["lineage"]["parent_run_id"]
            or parent_manifest.get("experiment_id") != "R002"
        ):
            raise ValueError(f"R004 parent manifest lineage mismatch: {path}")
        manifest["_parent_evaluations"] = parent_manifest["evaluations"]
        expected_updates = list(range(550, 2001, 50))
        if [item["global_update"] for item in manifest["evaluations"]] != [
            500,
            *expected_updates,
        ]:
            raise ValueError(f"R004 evaluation cadence mismatch: {path}")
        if [item["update"] for item in manifest["checkpoints"]] != expected_updates:
            raise ValueError(f"R004 checkpoint cadence mismatch: {path}")
        for checkpoint in manifest["checkpoints"]:
            for name in ("checkpoint", "probe_trajectory"):
                record = checkpoint[name]
                _verify_record(
                    manifest["_artifact_directory"] / Path(record["path"]).name,
                    record,
                )
        metrics_path = path.parent / "metrics.jsonl"
        _verify_record(metrics_path, manifest["metrics"])
        metrics = [
            json.loads(line) for line in metrics_path.read_text().splitlines() if line
        ]
        if [item["global_update"] for item in metrics] != list(range(501, 2001)):
            raise ValueError(f"R004 metric cadence mismatch: {path}")
        if (
            continuation_batch_hashes_sha256(metrics)
            != manifest["continuation_batch_hashes_sha256"]
        ):
            raise ValueError(f"R004 batch-stream digest mismatch: {path}")
        manifest["_metrics_path"] = metrics_path
        manifests.append(manifest)
    expected = {(seed, condition) for seed in range(16) for condition in CONDITIONS}
    actual = {(item["seed"], item["condition"]) for item in manifests}
    if len(manifests) != 64 or actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(f"R004 cohort incomplete; missing {missing}")
    return manifests


def _source_export(
    manifest: dict[str, Any], update: int, hardening: str
) -> tuple[Path, dict[str, Any]]:
    matches = [
        item
        for item in manifest["structural_exports"]
        if item["global_update"] == update and item["hardening"] == hardening
    ]
    if len(matches) != 1:
        raise ValueError(
            f"missing structural export: {manifest['run_id']}/{update}/{hardening}"
        )
    record = matches[0]["artifact"]
    if update == 500:
        path = Path(record["path"])
    else:
        path = manifest["_artifact_directory"] / Path(record["path"]).name
    _verify_record(path, record)
    return path, record


def adapt_parent_manifest(parent: Any, artifact_root: Path) -> dict[str, Any]:
    """Expose an immutable R002 update-500 parent through the R004 adapter."""
    return {
        "run_id": parent.manifest["run_id"],
        "seed": parent.manifest["seed"],
        "condition": parent.manifest["condition"],
        "representation": parent.manifest["representation"],
        "lineage": {"wiring_sha256": parent.manifest["pairing"]["wiring_sha256"]},
        "structural_exports": [
            {
                "global_update": 500,
                "hardening": hardening,
                "artifact": _artifact(path, f"parent_{hardening}_hard_circuit"),
            }
            for hardening, path in parent.exports.items()
        ],
        "_artifact_directory": artifact_root / parent.manifest["run_id"],
    }


def load_rule(
    manifest: dict[str, Any], update: int, hardening: str
) -> tuple[Any, dict[str, Any]]:
    path, record = _source_export(manifest, update, hardening)
    stored = profile.load_npz(path)
    gates = [stored[f"gate_{label}"].astype(np.uint8) for label in profile.LABELS]
    wires = [
        (stored[f"wire_{label}_a"], stored[f"wire_{label}_b"])
        for label in profile.LABELS
    ]
    if any(array.min() < 0 or array.max() > 15 for array in gates):
        raise ValueError("R004 gate ID outside the Boolean range")
    wiring_hash = profile.subset.tree_digest(
        [array for pair in wires for array in pair]
    )
    if wiring_hash != manifest["lineage"]["wiring_sha256"]:
        raise ValueError("R004 analysis wiring identity mismatch")
    dag = profile.refine.build(gates, wires)
    rule_hash = profile.subset.tree_digest(
        gates + [array for pair in wires for array in pair]
    )
    return dag, {
        "artifact": record,
        "rule_sha256": rule_hash,
        "wiring_sha256": wiring_hash,
        "training_update": update,
    }


def structural_record(
    manifest: dict[str, Any], update: int, hardening: str
) -> dict[str, Any]:
    dag, source = load_rule(manifest, update, hardening)
    summary = profile.compact_structure(profile.subset.dag_summary(dag))
    return {
        "run_id": manifest["run_id"],
        "seed": manifest["seed"],
        "condition": manifest["condition"],
        "representation": manifest["representation"],
        "global_update": update,
        "hardening": hardening,
        "source": source,
        **summary,
    }


def runtime_record(
    manifest: dict[str, Any],
    update: int,
    hardening: str,
    initializations: np.ndarray,
    labels: list[dict[str, Any]],
    target: np.ndarray,
    horizon: int,
    report_ticks: list[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dag, source = load_rule(manifest, update, hardening)
    trajectories, arrays, histories = profile.profile_trajectories(
        dag, initializations, labels, target, horizon
    )
    certificate = profile.universal_certificate(dag, target, horizon)
    groups = profile.group_summaries(trajectories, arrays, report_ticks)
    mixed = [
        (index, record)
        for index, record in enumerate(trajectories)
        if record["cycle_behavior"] == "some_phases_correct"
    ]
    counterexample = None
    if mixed:
        index, record = min(mixed, key=lambda item: item[1]["initialization_id"])
        period = int(record["cycle_period"])
        entry = int(record["cycle_entry_tick"])
        incorrect_phase = next(
            phase
            for phase in range(period)
            if phase not in set(record["correct_phase_indices"])
        )
        source_phase = (incorrect_phase - 20) % period
        packed = np.frombuffer(
            histories["full"][index][entry + source_phase], dtype=np.uint8
        )
        counterexample = {
            "initialization_id": record["initialization_id"],
            "cycle_entry_tick": entry,
            "cycle_period": period,
            "source_cycle_phase": source_phase,
            "initial_state_packed_hex": packed.tobytes().hex(),
        }
    for record in trajectories:
        record["orbit_certified_correct_onset"] = (
            record["observed_correct_suffix_start"]
            if record["cycle_behavior"] == "all_phases_correct"
            else None
        )
        record["universal_all_initial_state_onset"] = certificate[
            "certified_visible_settling_tick"
        ]
    return (
        {
            "run_id": manifest["run_id"],
            "seed": manifest["seed"],
            "condition": manifest["condition"],
            "representation": manifest["representation"],
            "global_update": update,
            "hardening": hardening,
            "source": source,
            "structure": profile.compact_structure(profile.subset.dag_summary(dag)),
            "initialization_groups": groups,
            "universal_state_certificate": certificate,
            "observed_phase_dependence": bool(mixed),
            "initialization_counterexample": counterexample,
            "universal_test_inconclusive": certificate["status"] == "inconclusive",
            "trajectories": trajectories,
        },
        arrays,
    )


def _binomial_two_sided(discordant_left: int, discordant_right: int) -> float:
    n = discordant_left + discordant_right
    if n == 0:
        return 1.0
    observed = min(discordant_left, discordant_right)
    tail = sum(math.comb(n, k) for k in range(observed + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _holm_two(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {
        ordered[0]: min(1.0, 2 * p_values[ordered[0]]),
        ordered[1]: max(min(1.0, 2 * p_values[ordered[0]]), p_values[ordered[1]]),
    }
    return adjusted


def _evaluation(manifest: dict[str, Any], update: int) -> dict[str, Any]:
    matches = [
        item for item in manifest["evaluations"] if item["global_update"] == update
    ]
    if len(matches) != 1:
        raise ValueError(f"missing R004 evaluation {manifest['run_id']}/{update}")
    return matches[0]


def _combined_evaluation(manifest: dict[str, Any], update: int) -> dict[str, Any]:
    if update <= 500:
        matches = [
            item for item in manifest["_parent_evaluations"] if item["update"] == update
        ]
        if len(matches) != 1:
            raise ValueError(
                f"missing R002 parent evaluation {manifest['run_id']}/{update}"
            )
        return matches[0]
    return _evaluation(manifest, update)


def aggregate(
    manifests: list[dict[str, Any]], structures: list[dict[str, Any]]
) -> dict[str, Any]:
    run_lookup = {(item["seed"], item["condition"]): item for item in manifests}
    structure_lookup = {
        (
            item["seed"],
            item["condition"],
            item["global_update"],
            item["hardening"],
        ): item
        for item in structures
    }
    endpoint_rows = []
    for manifest in manifests:
        baseline = _evaluation(manifest, 500)
        endpoint = _evaluation(manifest, 2000)
        before = bool(baseline["common"]["exact_all_32"])
        after = bool(endpoint["common"]["exact_all_32"])
        transition = (
            "gained_at_endpoint"
            if not before and after
            else "lost_at_endpoint"
            if before and not after
            else "exact_at_both_endpoints"
            if before
            else "inexact_at_both_endpoints"
        )
        history = [
            {
                "global_update": item["update"],
                "common_exact_all_32": bool(item["common"]["exact_all_32"]),
            }
            for item in manifest["_parent_evaluations"][:-1]
        ] + [
            {
                "global_update": item["global_update"],
                "common_exact_all_32": bool(item["common"]["exact_all_32"]),
            }
            for item in sorted(
                manifest["evaluations"], key=lambda item: item["global_update"]
            )
        ]
        successful_updates = [
            item["global_update"] for item in history if item["common_exact_all_32"]
        ]
        endpoint_rows.append(
            {
                "seed": manifest["seed"],
                "condition": manifest["condition"],
                "common_errors_500": baseline["common"]["bit_errors"],
                "common_errors_2000": endpoint["common"]["bit_errors"],
                "native_errors_500": baseline["native"]["bit_errors"],
                "native_errors_2000": endpoint["native"]["bit_errors"],
                "transition": transition,
                "success_timeline": history,
                "first_observed_exact_update": (
                    successful_updates[0] if successful_updates else None
                ),
                "ever_observed_exact": bool(successful_updates),
                "final_inexact_after_observed_success": bool(successful_updates)
                and not history[-1]["common_exact_all_32"],
                "never_observed_exact": not successful_updates,
            }
        )

    condition_summary = {}
    for condition in CONDITIONS:
        selected = [row for row in endpoint_rows if row["condition"] == condition]
        condition_summary[condition] = {
            "common_exact_500": sum(row["common_errors_500"] == 0 for row in selected),
            "common_exact_2000": sum(
                row["common_errors_2000"] == 0 for row in selected
            ),
            "native_exact_2000": sum(
                row["native_errors_2000"] == 0 for row in selected
            ),
            "history": {
                "ever_observed_exact": sum(
                    row["ever_observed_exact"] for row in selected
                ),
                "final_inexact_after_observed_success": sum(
                    row["final_inexact_after_observed_success"] for row in selected
                ),
                "never_observed_exact": sum(
                    row["never_observed_exact"] for row in selected
                ),
            },
            "transitions": {
                name: sum(row["transition"] == name for row in selected)
                for name in (
                    "gained_at_endpoint",
                    "lost_at_endpoint",
                    "exact_at_both_endpoints",
                    "inexact_at_both_endpoints",
                )
            },
        }

    p_values = {}
    mcnemar = {}
    for decay, categorical, truth in (
        (
            "reference_decay",
            "categorical_reference_decay",
            "truth_reference_decay",
        ),
        ("no_decay", "categorical_no_decay", "truth_no_decay"),
    ):
        categorical_only = truth_only = 0
        for seed in range(16):
            categorical_success = _evaluation(run_lookup[(seed, categorical)], 2000)[
                "common"
            ]["exact_all_32"]
            truth_success = _evaluation(run_lookup[(seed, truth)], 2000)["common"][
                "exact_all_32"
            ]
            categorical_only += int(categorical_success and not truth_success)
            truth_only += int(truth_success and not categorical_success)
        p_values[decay] = _binomial_two_sided(categorical_only, truth_only)
        mcnemar[decay] = {
            "categorical_only": categorical_only,
            "truth_only": truth_only,
            "p_value_two_sided_exact": p_values[decay],
        }
    adjusted = _holm_two(p_values)
    for name in mcnemar:
        mcnemar[name]["holm_adjusted_p_value"] = adjusted[name]

    generator = np.random.Generator(np.random.PCG64(20260911))
    indices = generator.integers(0, 16, size=(100000, 16), dtype=np.int64)
    bootstrap = {}
    for decay, categorical, truth in (
        ("reference_decay", "categorical_reference_decay", "truth_reference_decay"),
        ("no_decay", "categorical_no_decay", "truth_no_decay"),
    ):
        hard_changes = np.array(
            [
                (
                    _evaluation(run_lookup[(seed, truth)], 2000)["common"]["bit_errors"]
                    - _evaluation(run_lookup[(seed, truth)], 500)["common"][
                        "bit_errors"
                    ]
                )
                - (
                    _evaluation(run_lookup[(seed, categorical)], 2000)["common"][
                        "bit_errors"
                    ]
                    - _evaluation(run_lookup[(seed, categorical)], 500)["common"][
                        "bit_errors"
                    ]
                )
                for seed in range(16)
            ],
            dtype=np.float64,
        )
        size_gap_500 = np.array(
            [
                structure_lookup[(seed, truth, 500, "common")][
                    "visible_core_binary_nodes"
                ]
                - structure_lookup[(seed, categorical, 500, "common")][
                    "visible_core_binary_nodes"
                ]
                for seed in range(16)
            ],
            dtype=np.float64,
        )
        size_gap_2000 = np.array(
            [
                structure_lookup[(seed, truth, 2000, "common")][
                    "visible_core_binary_nodes"
                ]
                - structure_lookup[(seed, categorical, 2000, "common")][
                    "visible_core_binary_nodes"
                ]
                for seed in range(16)
            ],
            dtype=np.float64,
        )
        bootstrap[decay] = {}
        for name, values in (
            ("paired_hard_error_change_difference", hard_changes),
            ("size_gap_change", size_gap_2000 - size_gap_500),
        ):
            sampled = values[indices].mean(axis=1)
            bootstrap[decay][name] = {
                "raw_seed_values": values.tolist(),
                "mean": float(values.mean()),
                "percentile_95_interval": np.quantile(sampled, [0.025, 0.975]).tolist(),
            }
    return {
        "endpoint_rows": endpoint_rows,
        "condition_summary": condition_summary,
        "endpoint_coordinate_mcnemar": mcnemar,
        "paired_bootstrap": bootstrap,
        "bootstrap_identity": {
            "generator": "numpy.random.Generator(PCG64(20260911))",
            "shape": [100000, 16],
            "dtype": "int64",
            "indices_sha256": hashlib.sha256(indices.tobytes(order="C")).hexdigest(),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    if path.suffix == ".gz":
        with (
            path.open("wb") as output,
            gzip.GzipFile(
                filename="", mode="wb", compresslevel=9, mtime=0, fileobj=output
            ) as compressed,
        ):
            compressed.write(payload)
    else:
        path.write_bytes(payload)


def _summary_markdown(sweep_id: str, aggregate_result: dict[str, Any]) -> str:
    lines = [
        f"# R004 {sweep_id} analysis",
        "",
        "All endpoint results are continuations of the original R002 seed groups, not new trials.",
        "",
        "| Condition | Common exact at 500 | Common exact at 2,000 | Native exact at 2,000 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for condition, row in aggregate_result["condition_summary"].items():
        lines.append(
            f"| {condition} | {row['common_exact_500']}/16 | {row['common_exact_2000']}/16 | {row['native_exact_2000']}/16 |"
        )
    lines.extend(["", "## Endpoint transitions", ""])
    for condition, row in aggregate_result["condition_summary"].items():
        transitions = row["transitions"]
        lines.append(
            f"- `{condition}`: gained {transitions['gained_at_endpoint']}, lost "
            f"{transitions['lost_at_endpoint']}, exact at both "
            f"{transitions['exact_at_both_endpoints']}, inexact at both "
            f"{transitions['inexact_at_both_endpoints']}."
        )
    lines.extend(["", "## Complete saved-checkpoint history", ""])
    for condition, row in aggregate_result["condition_summary"].items():
        history = row["history"]
        lines.append(
            f"- `{condition}`: ever observed exact {history['ever_observed_exact']}, "
            f"later inexact at the endpoint {history['final_inexact_after_observed_success']}, "
            f"never observed exact {history['never_observed_exact']}."
        )
    lines.extend(["", "## Paired inference", ""])
    for decay, row in aggregate_result["endpoint_coordinate_mcnemar"].items():
        size = aggregate_result["paired_bootstrap"][decay]["size_gap_change"]
        errors = aggregate_result["paired_bootstrap"][decay][
            "paired_hard_error_change_difference"
        ]
        lines.append(
            f"- `{decay}`: exact paired McNemar p={row['p_value_two_sided_exact']:.6g}, "
            f"Holm-adjusted p={row['holm_adjusted_p_value']:.6g}; mean truth-minus-"
            f"categorical hard-error change difference {errors['mean']:.6g} "
            f"(95% paired bootstrap {errors['percentile_95_interval']}); mean size-gap "
            f"change {size['mean']:.6g} (95% paired bootstrap "
            f"{size['percentile_95_interval']})."
        )
    runtime = aggregate_result["runtime_summary"]
    lines.extend(
        [
            "",
            "## Runtime analysis",
            "",
            f"The fixed profiles account for {runtime['labeled_modes']} labeled modes and "
            f"{runtime['labeled_trajectories']} labeled trajectories. "
            f"Observed phase dependence occurs in {runtime['observed_phase_dependent_modes']} "
            f"modes, with explicit initialization counterexamples in "
            f"{runtime['initialization_counterexample_modes']}; "
            f"{runtime['universal_correct_modes']} universal tests certify all-state visible "
            f"settling, {runtime['universal_persistent_failure_modes']} certify persistent "
            f"visible failure, and {runtime['universal_inconclusive_modes']} are inconclusive. "
            f"Across labeled starts, {runtime['orbit_certified_correct_trajectories']} have "
            f"an orbit-certified correct onset, {runtime['mixed_phase_trajectories']} enter "
            f"mixed-phase cycles, {runtime['persistent_failure_trajectories']} enter cycles "
            f"with no correct phase, and {runtime['unresolved_trajectories']} remain unresolved "
            f"at the cap. Exactly {runtime['fixed_readout_all_66_exact_modes']} modes are "
            f"correct at tick 20 for all 66 fixed starts.",
            "",
            "Circuit counts are constructive FactoredDAG bounds, not minimum descriptions. "
            "Runtime, correctness, and circuit size are separate observables. The repeated "
            "checkpoints, probe grids, and two hardening modes are not independent trials. "
            "This fixed-grid continuation does not establish Kolmogorov complexity, grid-size "
            "generalization, or an infinite generator.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_series_svg(
    path: Path,
    title: str,
    x_values: list[int],
    series: dict[str, list[float]],
    y_label: str,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 25, 55, 65
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [value for values in series.values() for value in values]
    low = min(all_values, default=0.0)
    high = max(all_values, default=1.0)
    if high == low:
        high = low + 1
    x_min, x_max = min(x_values), max(x_values)
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea")

    def point(x: int, y: float) -> tuple[float, float]:
        px = left + plot_width * (x - x_min) / (x_max - x_min)
        py = top + plot_height * (high - y) / (high - low)
        return px, py

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111"/>',
        f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">{y_label}</text>',
        f'<text x="{left + plot_width / 2}" y="{height - 15}" text-anchor="middle" font-family="sans-serif" font-size="13">global optimizer update</text>',
    ]
    for index, (label, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = " ".join(
            f"{px:.2f},{py:.2f}"
            for px, py in map(lambda pair: point(*pair), zip(x_values, values))
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend_y = top + 18 * index
        lines.append(
            f'<line x1="{left + 15}" y1="{legend_y}" x2="{left + 35}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{left + 42}" y="{legend_y + 4}" font-family="sans-serif" font-size="11">{label}</text>'
        )
    for tick in (x_values[0], x_values[-1]):
        px, _ = point(tick, low)
        lines.append(
            f'<text x="{px}" y="{top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="11">{tick}</text>'
        )
    lines.append("</svg>\n")
    path.write_text("\n".join(lines))


def _render_bar_svg(
    path: Path, title: str, values: dict[str, float], y_label: str
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 25, 55, 115
    plot_width = width - left - right
    plot_height = height - top - bottom
    high = max(values.values(), default=1.0) or 1.0
    slot = plot_width / max(1, len(values))
    bar_width = slot * 0.62
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111"/>',
        f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">{y_label}</text>',
    ]
    for index, (label, value) in enumerate(values.items()):
        x = left + index * slot + (slot - bar_width) / 2
        bar_height = plot_height * value / high
        y = top + plot_height - bar_height
        lines.extend(
            (
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#2563eb"/>',
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 5:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.3g}</text>',
                f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height + 18}" transform="rotate(35 {x + bar_width / 2:.2f} {top + plot_height + 18})" text-anchor="start" font-family="sans-serif" font-size="10">{label}</text>',
            )
        )
    lines.append("</svg>\n")
    path.write_text("\n".join(lines))


def _render_xy_svg(
    path: Path,
    title: str,
    paths: list[tuple[str, list[tuple[float, float]]]],
    x_label: str,
    y_label: str,
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 80, 25, 55, 65
    plot_width, plot_height = width - left - right, height - top - bottom
    points = [point for _, values in paths for point in values]
    x_high = max((point[0] for point in points), default=1.0) or 1.0
    y_high = max((point[1] for point in points), default=1.0) or 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 15}" text-anchor="middle" font-family="sans-serif" font-size="13">{x_label}</text>',
        f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">{y_label}</text>',
    ]
    colors = {"categorical": "#2563eb", "truth": "#dc2626"}
    for label, values in paths:
        rendered = [
            (
                left + plot_width * x / x_high,
                top + plot_height * (y_high - y) / y_high,
            )
            for x, y in values
        ]
        color = colors[label]
        points_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in rendered)
        lines.append(
            f'<polyline points="{points_text}" fill="none" stroke="{color}" stroke-width="1" opacity="0.35"/>'
        )
        lines.extend(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="{color}"/>'
            for x, y in rendered
        )
    lines.append("</svg>\n")
    path.write_text("\n".join(lines))


def render_figures(
    result_directory: Path,
    manifests: list[dict[str, Any]],
    structures: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    aggregate_result: dict[str, Any],
) -> list[Path]:
    evaluation_updates = list(range(0, 2001, 50))
    errors = {
        condition: [
            float(
                np.mean(
                    [
                        _combined_evaluation(manifest, update)["common"]["bit_errors"]
                        for manifest in manifests
                        if manifest["condition"] == condition
                    ]
                )
            )
            for update in evaluation_updates
        ]
        for condition in CONDITIONS
    }
    training_path = result_directory / "paired_error_trajectories.svg"
    _render_series_svg(
        training_path,
        "R004 common-hard error trajectories",
        evaluation_updates,
        errors,
        "mean visible bit errors over 16 seeds",
    )
    soft_losses = {
        condition: [
            float(
                np.mean(
                    [
                        _combined_evaluation(manifest, update)[
                            "soft_terminal_summed_squared_error"
                        ]
                        for manifest in manifests
                        if manifest["condition"] == condition
                    ]
                )
            )
            for update in evaluation_updates
        ]
        for condition in CONDITIONS
    }
    soft_path = result_directory / "paired_soft_loss_trajectories.svg"
    _render_series_svg(
        soft_path,
        "R004 soft-loss trajectories",
        evaluation_updates,
        soft_losses,
        "mean terminal summed squared error over 16 seeds",
    )

    transition_path = result_directory / "endpoint_transitions.svg"
    _render_bar_svg(
        transition_path,
        "R004 endpoint exactness transitions",
        {
            f"{condition}:{transition}": count
            for condition, summary in aggregate_result["condition_summary"].items()
            for transition, count in summary["transitions"].items()
        },
        "training seeds",
    )

    structure_lookup = {
        (row["seed"], row["condition"], row["global_update"]): row
        for row in structures
        if row["hardening"] == "common"
    }
    structure_updates = [500, 750, 1000, 1250, 1500, 1750, 2000]
    gaps = {}
    for decay, categorical, truth in (
        ("reference_decay", "categorical_reference_decay", "truth_reference_decay"),
        ("no_decay", "categorical_no_decay", "truth_no_decay"),
    ):
        gaps[decay] = [
            float(
                np.mean(
                    [
                        structure_lookup[(seed, truth, update)][
                            "visible_core_binary_nodes"
                        ]
                        - structure_lookup[(seed, categorical, update)][
                            "visible_core_binary_nodes"
                        ]
                        for seed in range(16)
                    ]
                )
            )
            for update in structure_updates
        ]
    size_path = result_directory / "size_gap_trajectory.svg"
    _render_series_svg(
        size_path,
        "R004 truth minus categorical visible-core size",
        structure_updates,
        gaps,
        "mean paired constructive operation gap",
    )

    manifest_lookup = {
        (manifest["seed"], manifest["condition"]): manifest for manifest in manifests
    }
    joint_paths = [
        (
            CONDITIONS[condition].representation,
            [
                (
                    structure_lookup[(seed, condition, update)][
                        "visible_core_binary_nodes"
                    ],
                    _evaluation(manifest_lookup[(seed, condition)], update)["common"][
                        "bit_errors"
                    ],
                )
                for update in structure_updates
            ],
        )
        for seed in range(16)
        for condition in CONDITIONS
    ]
    joint_path = result_directory / "joint_size_error_paths.svg"
    _render_xy_svg(
        joint_path,
        "R004 common-hard size and error paths",
        joint_paths,
        "constructive visible-core binary nodes",
        "visible bit errors",
    )

    runtime_path = result_directory / "runtime_cycle_evidence.svg"
    _render_bar_svg(
        runtime_path,
        "R004 runtime and universal evidence",
        {
            "phase-dependent": sum(
                row["observed_phase_dependence"] for row in runtime_rows
            ),
            "universal-inconclusive": sum(
                row["universal_test_inconclusive"] for row in runtime_rows
            ),
            "universally-correct": sum(
                row["universal_state_certificate"]["status"]
                == "all_initial_states_visible_target_certified"
                for row in runtime_rows
            ),
            "persistent-failure": sum(
                row["universal_state_certificate"]["status"]
                == "persistent_visible_failure_certified"
                for row in runtime_rows
            ),
        },
        "labeled run, update, and hardening modes",
    )

    cost_path = result_directory / "training_cost.svg"
    _render_bar_svg(
        cost_path,
        "R004 synchronized training cost",
        {
            condition: float(
                np.mean(
                    [
                        manifest["synchronized_update_seconds"]
                        for manifest in manifests
                        if manifest["condition"] == condition
                    ]
                )
            )
            for condition in CONDITIONS
        },
        "mean synchronized seconds per continued run",
    )
    return [
        training_path,
        soft_path,
        transition_path,
        size_path,
        joint_path,
        runtime_path,
        cost_path,
    ]


def run_analysis(
    config_path: Path,
    sweep_id: str,
    analysis_id: str,
    result_directory: Path,
    output_artifact_root: Path,
    source_artifact_root: Path,
) -> dict[str, Any]:
    """Execute the fixed R004 offline analysis against completed descendants."""
    if result_directory.exists() or output_artifact_root.exists():
        raise FileExistsError("R004 analysis output already exists")
    started = time.perf_counter()
    repository_root = config_path.resolve().parents[2]
    config_bytes = config_path.read_bytes()
    document = json.loads(config_bytes)
    validate_config_contract(document)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    manifests = enumerate_runs(
        repository_root, source_artifact_root, sweep_id, config_sha256
    )
    result_directory.mkdir(parents=True)
    output_artifact_root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": "R004",
        "kind": "R004_structural_runtime_endpoint_analysis",
        "analysis_id": analysis_id,
        "sweep_id": sweep_id,
        "status": "running",
        "config_sha256": config_sha256,
        "analysis_code": code_provenance(),
        "source_run_count": len(manifests),
        "failures": [],
    }
    manifest_path = result_directory / "manifest.json"
    write_json(manifest_path, manifest)
    try:
        original_path = repository_root / document["evaluation"]["original_probe_path"]
        fresh_path = (
            repository_root / document["runtime_profile"]["additional_probe_path"]
        )
        if (
            sha256_file(original_path)
            != document["evaluation"]["original_probe_sha256"]
        ):
            raise ValueError("R004 original probe identity mismatch")
        if (
            sha256_file(fresh_path)
            != document["runtime_profile"]["additional_probe_sha256"]
        ):
            raise ValueError("R004 reused additional probe identity mismatch")
        original = profile.load_npz(original_path)
        fresh = profile.load_npz(fresh_path)["inputs"].astype(np.uint8)
        original_inputs = original["inputs"].astype(np.uint8)
        target = original["target"][0, ..., 0].astype(np.uint8)
        initializations, labels = profile.make_initializations(original_inputs, fresh)
        profile.refine.validate_refinement()
        cycle_checks = profile.validate_cycle_bookkeeping()
        validation_path = result_directory / "validation.json"
        write_json(
            validation_path,
            {
                "status": "passed",
                "source_runs_verified": len(manifests),
                "original_probe": _artifact(original_path, "R001_fixed_probe"),
                "reused_additional_probe": _artifact(
                    fresh_path, "R002_reused_additional_probe"
                ),
                "factored_dag_algebra": "passed",
                "cycle_bookkeeping": cycle_checks,
                "source_checkpoint_and_export_identity": "verified_by_run_manifests_and_analysis_loader",
            },
        )

        structures = [
            structural_record(manifest_source, update, hardening)
            for manifest_source in manifests
            for update in document["structure"]["optimizer_updates"]
            for hardening in document["structure"]["hardening_modes"]
        ]
        structures_path = result_directory / "structure.jsonl"
        _write_jsonl(structures_path, structures)

        runtime_rows = []
        cache: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
        array_records: dict[str, dict[str, Any]] = {}
        for manifest_source in manifests:
            for update in document["runtime_profile"]["optimizer_updates"]:
                for hardening in document["runtime_profile"]["hardening_modes"]:
                    dag, source = load_rule(manifest_source, update, hardening)
                    cache_key = source["rule_sha256"]
                    if cache_key not in cache:
                        cache[cache_key] = runtime_record(
                            manifest_source,
                            update,
                            hardening,
                            initializations,
                            labels,
                            target,
                            document["runtime_profile"]["horizon"],
                            document["runtime_profile"]["report_ticks"],
                        )
                        array_path = output_artifact_root / (
                            f"runtime_rule_{cache_key}.npz"
                        )
                        with array_path.open("wb") as handle:
                            np.savez_compressed(handle, **cache[cache_key][1])
                        array_records[cache_key] = _artifact(
                            array_path, "R004_unique_rule_runtime_arrays"
                        )
                    base_record, _arrays = cache[cache_key]
                    record = {
                        **base_record,
                        "run_id": manifest_source["run_id"],
                        "seed": manifest_source["seed"],
                        "condition": manifest_source["condition"],
                        "representation": manifest_source["representation"],
                        "global_update": update,
                        "hardening": hardening,
                        "source": source,
                        "reused_rule_profile": base_record["run_id"]
                        != manifest_source["run_id"]
                        or base_record["global_update"] != update
                        or base_record["hardening"] != hardening,
                    }
                    record["arrays"] = array_records[cache_key]
                    runtime_rows.append(record)
        runtime_path = result_directory / "runtime_profiles.jsonl.gz"
        _write_jsonl(runtime_path, runtime_rows)

        aggregate_result = aggregate(manifests, structures)
        aggregate_result["runtime_summary"] = {
            "labeled_modes": len(runtime_rows),
            "labeled_trajectories": len(runtime_rows) * 66,
            "unique_rule_profiles": len(cache),
            "observed_phase_dependent_modes": sum(
                row["observed_phase_dependence"] for row in runtime_rows
            ),
            "universal_inconclusive_modes": sum(
                row["universal_test_inconclusive"] for row in runtime_rows
            ),
            "initialization_counterexample_modes": sum(
                row["initialization_counterexample"] is not None for row in runtime_rows
            ),
            "universal_correct_modes": sum(
                row["universal_state_certificate"]["status"]
                == "all_initial_states_visible_target_certified"
                for row in runtime_rows
            ),
            "universal_persistent_failure_modes": sum(
                row["universal_state_certificate"]["status"]
                == "persistent_visible_failure_certified"
                for row in runtime_rows
            ),
            "fixed_readout_all_66_exact_modes": sum(
                all(
                    trajectory["errors_at_20"] == 0
                    for trajectory in row["trajectories"]
                )
                for row in runtime_rows
            ),
            "orbit_certified_correct_trajectories": sum(
                trajectory["orbit_certified_correct_onset"] is not None
                for row in runtime_rows
                for trajectory in row["trajectories"]
            ),
            "mixed_phase_trajectories": sum(
                trajectory["cycle_behavior"] == "some_phases_correct"
                for row in runtime_rows
                for trajectory in row["trajectories"]
            ),
            "persistent_failure_trajectories": sum(
                trajectory["cycle_behavior"] == "no_phases_correct"
                for row in runtime_rows
                for trajectory in row["trajectories"]
            ),
            "unresolved_trajectories": sum(
                trajectory["cycle_behavior"] == "unresolved_by_cap"
                for row in runtime_rows
                for trajectory in row["trajectories"]
            ),
        }
        analysis_path = result_directory / "analysis.json"
        write_json(analysis_path, aggregate_result)
        summary_path = result_directory / "summary.md"
        summary_path.write_text(_summary_markdown(sweep_id, aggregate_result))
        figure_paths = render_figures(
            result_directory,
            manifests,
            structures,
            runtime_rows,
            aggregate_result,
        )
        manifest.update(
            {
                "status": "completed",
                "ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "wall_seconds": time.perf_counter() - started,
                "structure_snapshot_count": len(structures),
                "runtime_mode_count": len(runtime_rows),
                "runtime_trajectory_count": len(runtime_rows) * 66,
                "artifacts": [
                    _artifact(structures_path, "R004_structure_table"),
                    _artifact(runtime_path, "R004_runtime_profile_table"),
                    _artifact(analysis_path, "R004_aggregate_analysis"),
                    _artifact(summary_path, "R004_analysis_summary"),
                    _artifact(validation_path, "R004_analysis_validation"),
                    *[_artifact(path, "R004_analysis_figure") for path in figure_paths],
                ],
            }
        )
        write_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["failure_reason"] = f"{type(error).__name__}: {error}"
        write_json(manifest_path, manifest)
        raise
    return manifest
