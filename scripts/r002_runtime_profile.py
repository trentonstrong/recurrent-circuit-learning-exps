#!/usr/bin/env python3
"""Bounded post-hoc runtime profile of the frozen R002 Boolean circuits."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import gzip
import hashlib
import json
import multiprocessing
import os
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from scripts.build_release_assets import build_bundle

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = ROOT / "reviews" / "R002_circuits"
sys.path.insert(0, str(REVIEW_DIR))
import analyze_subset as subset  # noqa: E402
import refine_circuits as refine  # noqa: E402

LABELS = subset.LABELS
BIT_WEIGHTS = np.array([8, 4, 2, 1], dtype=np.uint8)
EXPECTED_CONDITIONS = (
    "categorical_reference_decay",
    "truth_reference_decay",
    "categorical_no_decay",
    "truth_no_decay",
)
INITIALIZATION_GROUPS = ("original_probe", "fresh_probe", "all_zero", "all_one")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def file_record(path: Path, root: Path = ROOT) -> dict[str, Any]:
    try:
        rendered = path.relative_to(root).as_posix()
    except ValueError:
        rendered = str(path)
    return {
        "path": rendered,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def code_provenance() -> dict[str, Any]:
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    status = git_output("status", "--short").splitlines()
    return {
        "commit": git_output("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status,
        "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
    }


def cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def validate_config(config: dict[str, Any]) -> None:
    expected = {
        "protocol_id": "R002_runtime_profile_v1",
        "source_cohort": "R002_formal_attempt01",
        "training_code_commit": "d220b65da822707cd7db3dc90df69d9250aa0f77",
        "optimizer_update": 500,
        "hardening_modes": ["common", "native"],
        "grid_size": [16, 16],
        "channels": 8,
        "boundary": "constant_zero",
        "observer_channel": 0,
        "target": "anchored_2x2_checkerboard",
        "horizon": 256,
        "report_ticks": [20, 21, 22, 25, 32, 64, 128, 256],
        "original_probe_sha256": (
            "d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f"
        ),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"invalid frozen configuration field {key!r}")
    initial = config["initializations"]
    if initial != {
        "original_probe_count": 32,
        "fresh_probe_count": 32,
        "fresh_probe_bit_generator": "numpy.random.PCG64",
        "fresh_probe_seed": 20260910,
        "all_zero_count": 1,
        "all_one_count": 1,
    }:
        raise ValueError("invalid frozen initialization configuration")
    if config["universal_state_diagnostic"] != {
        "method": "sound_set_valued_boolean_propagation",
        "tick_cap": 256,
    }:
        raise ValueError("invalid universal-state diagnostic configuration")


def source_reference(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    references = list(manifest["artifacts"])
    for checkpoint in manifest["checkpoints"]:
        references.extend((checkpoint["checkpoint"], checkpoint["probe_trajectory"]))
    matches = [record for record in references if Path(record["path"]).name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one source reference for {name}")
    return matches[0]


def verify_file(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_record(path)
    if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
        raise ValueError(f"artifact identity mismatch: {path}")
    return actual


def enumerate_cohort(artifact_root: Path) -> list[dict[str, Any]]:
    manifests = []
    for path in sorted((ROOT / "results").glob("R002_formal_attempt01_seed*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["status"] != "completed" or manifest["current_update"] != 500:
            raise ValueError(f"incomplete source run: {manifest['run_id']}")
        if manifest["code"]["commit"] != "d220b65da822707cd7db3dc90df69d9250aa0f77":
            raise ValueError(f"unexpected training commit: {manifest['run_id']}")
        manifest["_manifest_path"] = path
        manifest["_artifact_directory"] = artifact_root / manifest["run_id"]
        manifests.append(manifest)
    members = {(m["seed"], m["condition"]) for m in manifests}
    expected = {(seed, condition) for seed in range(16) for condition in EXPECTED_CONDITIONS}
    if len(manifests) != 64 or members != expected:
        raise ValueError("source cohort is not the complete 16 by 4 design")
    return manifests


def load_rule(
    manifest: dict[str, Any], hardening: str, artifact_root: Path
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], Any, dict[str, Any]]:
    run_id = manifest["run_id"]
    directory = artifact_root / run_id
    export_name = f"final_{hardening}_circuit.npz"
    checkpoint_name = "checkpoint_update_500.npz"
    export_path = directory / export_name
    checkpoint_path = directory / checkpoint_name
    export_identity = verify_file(export_path, source_reference(manifest, export_name))
    checkpoint_identity = verify_file(
        checkpoint_path, source_reference(manifest, checkpoint_name)
    )
    export = load_npz(export_path)
    checkpoint = load_npz(checkpoint_path)
    metadata = json.loads(str(checkpoint["metadata_json"]))
    if (
        int(checkpoint["format_version"]) != 2
        or int(checkpoint["update_index"]) != 500
        or metadata["condition"] != manifest["condition"]
        or metadata["representation"] != manifest["representation"]
        or metadata["seed"] != manifest["seed"]
        or metadata["config_sha256"] != manifest["config_sha256"]
        or metadata["wiring_sha256"] != manifest["pairing"]["wiring_sha256"]
    ):
        raise ValueError(f"checkpoint metadata mismatch: {run_id}")

    gates = [export[f"gate_{label}"].astype(np.uint8) for label in LABELS]
    wires = [
        (
            export[f"wire_{label}_a"].copy(),
            export[f"wire_{label}_b"].copy(),
        )
        for label in LABELS
    ]
    if any(g.min() < 0 or g.max() > 15 for g in gates):
        raise ValueError(f"gate ID out of range: {run_id}/{hardening}")
    upper = 9
    for index in range(3):
        a, b = wires[index]
        if gates[index].shape[-1] != len(a) or a.min() < 0 or b.min() < 0:
            raise ValueError(f"perception array shape mismatch: {run_id}")
        if a.max() >= upper or b.max() >= upper:
            raise ValueError(f"perception wiring index out of range: {run_id}")
        upper = len(a)
    upper = 8 + 16 * 8 * len(wires[2][0])
    for index in range(3, len(LABELS)):
        a, b = wires[index]
        if gates[index].shape != a.shape or a.shape != b.shape:
            raise ValueError(f"update array shape mismatch: {run_id}")
        if a.min() < 0 or b.min() < 0 or a.max() >= upper or b.max() >= upper:
            raise ValueError(f"update wiring index out of range: {run_id}")
        upper = len(a)
    if upper != 8:
        raise ValueError(f"final circuit does not have eight channels: {run_id}")
    if subset.tree_digest([array for pair in wires for array in pair]) != manifest["pairing"]["wiring_sha256"]:
        raise ValueError(f"wiring hash mismatch: {run_id}")

    _, decoded = subset.decode(checkpoint, manifest["representation"])
    if any(not np.array_equal(left, right) for left, right in zip(gates, decoded[hardening])):
        raise ValueError(f"export/checkpoint extraction mismatch: {run_id}/{hardening}")
    dag = refine.build(gates, wires)
    rule_hash = subset.tree_digest(gates + [array for pair in wires for array in pair])
    identities = {
        "export": export_identity,
        "checkpoint": checkpoint_identity,
        "rule_sha256": rule_hash,
    }
    return gates, wires, dag, identities


def make_initializations(
    original_probe: np.ndarray, fresh_probe: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    zeros = np.zeros((1, 16, 16, 8), dtype=np.uint8)
    ones = np.ones((1, 16, 16, 8), dtype=np.uint8)
    arrays = np.concatenate((original_probe, fresh_probe, zeros, ones), axis=0)
    labels = []
    for group, count in (("original_probe", 32), ("fresh_probe", 32)):
        labels.extend(
            {"initialization_id": f"{group}_{index:02d}", "group": group, "index": index}
            for index in range(count)
        )
    labels.append({"initialization_id": "all_zero", "group": "all_zero", "index": 0})
    labels.append({"initialization_id": "all_one", "group": "all_one", "index": 0})
    return arrays, labels


def summarize_orbit(
    full_states: list[bytes], visible_states: list[bytes], errors: list[int]
) -> dict[str, Any]:
    seen: dict[bytes, int] = {}
    cycle_entry = None
    cycle_period = None
    for tick, state in enumerate(full_states):
        if state in seen:
            cycle_entry = seen[state]
            cycle_period = tick - cycle_entry
            break
        seen[state] = tick
    hits = [tick for tick, error in enumerate(errors) if error == 0]
    suffix_start = None
    if errors[-1] == 0:
        suffix_start = max((tick for tick, error in enumerate(errors) if error != 0), default=-1) + 1
    result: dict[str, Any] = {
        "errors_at_20": errors[20] if len(errors) > 20 else None,
        "first_hit_tick": hits[0] if hits else None,
        "observed_correct_suffix_start": suffix_start,
        "observed_correct_suffix_length": len(errors) - suffix_start if suffix_start is not None else 0,
        "cycle_entry_tick": cycle_entry,
        "cycle_period": cycle_period,
        "cycle_target_fraction": None,
        "cycle_min_errors": None,
        "cycle_mean_errors": None,
        "cycle_max_errors": None,
        "cycle_behavior": "unresolved_by_cap",
        "correct_phase_indices": [],
        "phase_at_tick_20": None,
        "visible_cycle_period": None,
        "cycle_signature_sha256": None,
        "cycle_phase_offset_from_canonical": None,
        "evidence": {
            "finite_replay_through_cap": True,
            "exact_full_state_recurrence": cycle_entry is not None,
            "universal_state_certificate": "reported_per_circuit",
        },
    }
    if cycle_entry is None or cycle_period is None:
        return result
    cycle_full = full_states[cycle_entry : cycle_entry + cycle_period]
    cycle_visible = visible_states[cycle_entry : cycle_entry + cycle_period]
    cycle_errors = errors[cycle_entry : cycle_entry + cycle_period]
    correct = [index for index, error in enumerate(cycle_errors) if error == 0]
    fraction = len(correct) / cycle_period
    if fraction == 1:
        behavior = "all_phases_correct"
    elif fraction == 0:
        behavior = "no_phases_correct"
    else:
        behavior = "some_phases_correct"
    canonical_start = min(range(cycle_period), key=cycle_full.__getitem__)
    canonical = cycle_full[canonical_start:] + cycle_full[:canonical_start]
    visible_period = cycle_period
    for divisor in range(1, cycle_period + 1):
        if cycle_period % divisor == 0 and all(
            cycle_visible[index] == cycle_visible[index % divisor]
            for index in range(cycle_period)
        ):
            visible_period = divisor
            break
    result.update(
        {
            "cycle_target_fraction": fraction,
            "cycle_min_errors": min(cycle_errors),
            "cycle_mean_errors": sum(cycle_errors) / cycle_period,
            "cycle_max_errors": max(cycle_errors),
            "cycle_behavior": behavior,
            "correct_phase_indices": correct,
            "phase_at_tick_20": (
                (20 - cycle_entry) % cycle_period if 20 >= cycle_entry else None
            ),
            "visible_cycle_period": visible_period,
            "cycle_signature_sha256": hashlib.sha256(b"".join(canonical)).hexdigest(),
            "cycle_phase_offset_from_canonical": (-canonical_start) % cycle_period,
        }
    )
    return result


def validate_cycle_bookkeeping() -> dict[str, Any]:
    fixed = summarize_orbit([b"a", b"b", b"c", b"c"], [b"0"] * 4, [1, 1, 0, 0])
    two = summarize_orbit([b"a", b"b", b"a"], [b"0", b"1", b"0"], [0, 1, 0])
    hidden = summarize_orbit([b"a", b"b", b"a"], [b"0", b"0", b"0"], [0, 0, 0])
    unresolved = summarize_orbit([b"a", b"b", b"c"], [b"0", b"1", b"0"], [1, 1, 0])
    assert (fixed["cycle_entry_tick"], fixed["cycle_period"]) == (2, 1)
    assert (two["cycle_entry_tick"], two["cycle_period"], two["visible_cycle_period"]) == (0, 2, 2)
    assert hidden["cycle_period"] == 2 and hidden["visible_cycle_period"] == 1
    assert unresolved["cycle_entry_tick"] is None
    assert unresolved["observed_correct_suffix_start"] == 2
    assert unresolved["observed_correct_suffix_length"] == 1
    return {
        "transient_fixed_point": True,
        "two_cycle": True,
        "hidden_cycle_constant_visible": True,
        "unresolved_by_cap": True,
        "late_single_correct_frame_is_finite_suffix_only": True,
    }


def universal_certificate(dag: Any, target: np.ndarray, cap: int) -> dict[str, Any]:
    state = np.full((16, 16, 8), 3, dtype=np.uint8)
    counts = []
    invariant_tick = None
    for tick in range(cap + 1):
        visible = state[..., 0]
        known = visible != 3
        wrong = known & ((visible == 2) != target)
        counts.append(
            {
                "tick": tick,
                "known_correct": int(np.sum(known & ~wrong)),
                "known_wrong": int(np.sum(wrong)),
                "unknown": int(np.sum(~known)),
            }
        )
        if tick == cap:
            break
        next_state = subset.abstract_step(state, dag)
        if np.array_equal(next_state, state):
            invariant_tick = tick
            break
        state = next_state
    correct_ticks = [row["tick"] for row in counts if row["known_correct"] == 256]
    settling = None
    if invariant_tick is not None and counts[-1]["known_correct"] == 256:
        for tick in correct_ticks:
            if all(row["known_correct"] == 256 for row in counts[tick:]):
                settling = tick
                break
    if settling is not None:
        status = "all_initial_states_visible_target_certified"
    elif invariant_tick is not None and counts[-1]["known_wrong"] > 0:
        status = "persistent_visible_failure_certified"
    else:
        status = "inconclusive"
    return {
        "method": "exact_set_images_with_factored_boolean_dag",
        "tick_cap": cap,
        "counts_by_tick": counts,
        "invariant_abstraction_tick": invariant_tick,
        "status": status,
        "certified_visible_settling_tick": settling,
    }


def pack_rows(state: np.ndarray) -> list[bytes]:
    packed = np.packbits(state.reshape(state.shape[0], -1), axis=1, bitorder="big")
    return [row.tobytes() for row in packed]


def profile_trajectories(
    dag: Any,
    initializations: np.ndarray,
    labels: list[dict[str, Any]],
    target: np.ndarray,
    horizon: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, list[bytes]]]:
    count = len(initializations)
    errors = np.zeros((count, horizon + 1), dtype=np.uint16)
    visible_changes = np.zeros((count, horizon + 1), dtype=np.uint16)
    full_changes = np.zeros((count, horizon + 1), dtype=np.uint16)
    full_histories = [[] for _ in range(count)]
    visible_histories = [[] for _ in range(count)]
    state = initializations.copy()
    previous = None
    for tick in range(horizon + 1):
        errors[:, tick] = np.sum(state[..., 0] != target, axis=(1, 2))
        if previous is not None:
            visible_changes[:, tick] = np.sum(
                state[..., 0] != previous[..., 0], axis=(1, 2)
            )
            full_changes[:, tick] = np.sum(state != previous, axis=(1, 2, 3))
        packed_full = pack_rows(state)
        packed_visible = pack_rows(state[..., :1])
        for index in range(count):
            full_histories[index].append(packed_full[index])
            visible_histories[index].append(packed_visible[index])
        if tick < horizon:
            previous = state
            state = dag.evaluate(subset.base.patches(state))
    records = []
    for index, label in enumerate(labels):
        record = dict(label)
        record.update(
            summarize_orbit(
                full_histories[index],
                visible_histories[index],
                errors[index].astype(int).tolist(),
            )
        )
        records.append(record)
    arrays = {
        "errors": errors,
        "visible_hamming_changes": visible_changes,
        "full_state_hamming_changes": full_changes,
    }
    histories = {"full": full_histories, "visible": visible_histories}
    return records, arrays, histories


def group_summaries(
    records: list[dict[str, Any]], arrays: dict[str, np.ndarray], report_ticks: list[int]
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["group"]].append(index)
    result = {}
    for group, indices in grouped.items():
        group_errors = arrays["errors"][indices]
        suffixes = [records[index]["observed_correct_suffix_start"] for index in indices]
        cycles = [records[index]["cycle_signature_sha256"] for index in indices]
        result[group] = {
            "initialization_count": len(indices),
            "aggregate_errors_by_tick": group_errors.sum(axis=0).astype(int).tolist(),
            "perfect_initializations_by_tick": (group_errors == 0).sum(axis=0).astype(int).tolist(),
            "first_simultaneous_hit_tick": next(
                (tick for tick in range(group_errors.shape[1]) if np.all(group_errors[:, tick] == 0)),
                None,
            ),
            "all_have_observed_correct_suffix": all(value is not None for value in suffixes),
            "latest_observed_correct_suffix_start": (
                max(suffixes) if all(value is not None for value in suffixes) else None
            ),
            "known_distinct_full_cycles": len({cycle for cycle in cycles if cycle is not None}),
            "unresolved_trajectory_count": sum(cycle is None for cycle in cycles),
            "report_ticks": {
                str(tick): {
                    "aggregate_errors": int(group_errors[:, tick].sum()),
                    "perfect_initializations": int(np.sum(group_errors[:, tick] == 0)),
                    "all_correct": bool(np.all(group_errors[:, tick] == 0)),
                }
                for tick in report_ticks
            },
        }
    return result


def compact_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in certificate.items() if key != "counts_by_tick"}


def compact_group_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key not in {"aggregate_errors_by_tick", "perfect_initializations_by_tick"}
    }


def compact_structure(structure: dict[str, Any]) -> dict[str, Any]:
    return {
        key: structure[key]
        for key in (
            "binary_nodes_all_channels",
            "and_nodes",
            "xor_nodes",
            "visible_recurrent_channels",
            "visible_core_binary_nodes",
            "relevant_local_inputs",
        )
    }


def make_phase_witness(
    run_id: str,
    hardening: str,
    record: dict[str, Any],
    history: list[bytes],
    dag: Any,
    target: np.ndarray,
) -> dict[str, Any]:
    period = int(record["cycle_period"])
    entry = int(record["cycle_entry_tick"])
    incorrect_phase = next(
        phase
        for phase in range(period)
        if phase not in set(record["correct_phase_indices"])
    )
    source_phase = (incorrect_phase - 20) % period
    packed = np.frombuffer(history[entry + source_phase], dtype=np.uint8).copy()
    state = np.unpackbits(packed, bitorder="big")[: 16 * 16 * 8].reshape(1, 16, 16, 8)
    replay = subset.hard_rollout(state, dag, 20)
    error = int(np.sum(replay[-1, 0, ..., 0] != target))
    cycle_errors = record.pop("_cycle_errors")
    expected_error = int(cycle_errors[incorrect_phase])
    if error != expected_error:
        raise ValueError(f"phase witness replay mismatch: {run_id}/{hardening}")
    metadata = {
        "run_id": run_id,
        "hardening": hardening,
        "initialization_id": record["initialization_id"],
        "codec": "C-order flattened Boolean bits, numpy.packbits bitorder=big",
        "cycle_entry_tick": entry,
        "cycle_period": period,
        "incorrect_target_phase": incorrect_phase,
        "source_cycle_phase": source_phase,
        "source_orbit_tick": entry + source_phase,
        "visible_errors_after_20_ticks": error,
    }
    return {**metadata, "initial_state_packed_hex": packed.tobytes().hex()}


def write_phase_witness(
    path: Path, payload: dict[str, Any], run_id: str, hardening: str
) -> dict[str, Any]:
    payload = dict(payload)
    packed = np.frombuffer(
        bytes.fromhex(payload.pop("initial_state_packed_hex")), dtype=np.uint8
    ).copy()
    payload["run_id"] = run_id
    payload["hardening"] = hardening
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, initial_state_packed=packed, metadata_json=json.dumps(payload))
    return {**payload, "artifact": file_record(path)}


def profile_mode(
    manifest: dict[str, Any],
    hardening: str,
    artifact_root: Path,
    initializations: np.ndarray,
    labels: list[dict[str, Any]],
    target: np.ndarray,
    config: dict[str, Any],
    output_artifact_root: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    started = time.perf_counter()
    gates, wires, dag, identities = load_rule(manifest, hardening, artifact_root)
    records, arrays, histories = profile_trajectories(
        dag, initializations, labels, target, config["horizon"]
    )
    for index, record in enumerate(records):
        if record["cycle_entry_tick"] is not None:
            entry = record["cycle_entry_tick"]
            period = record["cycle_period"]
            record["_cycle_errors"] = arrays["errors"][index, entry : entry + period].astype(int).tolist()
    certificate = universal_certificate(dag, target, config["horizon"])
    groups = group_summaries(records, arrays, config["report_ticks"])
    phase_witness = None
    mixed = [
        (index, record)
        for index, record in enumerate(records)
        if record["cycle_behavior"] == "some_phases_correct"
    ]
    if mixed:
        index, record = min(mixed, key=lambda item: item[1]["initialization_id"])
        phase_witness = make_phase_witness(
            manifest["run_id"],
            hardening,
            record,
            histories["full"][index],
            dag,
            target,
        )
    for record in records:
        record.pop("_cycle_errors", None)
        record["certified_visible_settling_tick"] = certificate[
            "certified_visible_settling_tick"
        ]
    structure = subset.dag_summary(dag)
    return (
        {
            "run_id": manifest["run_id"],
            "seed": manifest["seed"],
            "condition": manifest["condition"],
            "representation": manifest["representation"],
            "hardening": hardening,
            "source": identities,
            "structure": structure,
            "universal_state_certificate": certificate,
            "initialization_groups": groups,
            "trajectories": records,
            "phase_witnesses": [],
            "_phase_witness": phase_witness,
            "profile_seconds": time.perf_counter() - started,
        },
        arrays,
    )


def profile_mode_task(arguments: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    return profile_mode(*arguments)


def profile_cache_paths(cache_directory: Path, rule_sha256: str) -> tuple[Path, Path]:
    return (
        cache_directory / f"{rule_sha256}.json",
        cache_directory / f"{rule_sha256}.npz",
    )


def load_profile_cache(
    cache_directory: Path,
    rule_sha256: str,
    config_sha256: str,
    initializations_sha256: str,
    implementation_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    record_path, arrays_path = profile_cache_paths(cache_directory, rule_sha256)
    if not record_path.exists() or not arrays_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if (
        record.get("schema_version") != 1
        or record.get("rule_sha256") != rule_sha256
        or record.get("config_sha256") != config_sha256
        or record.get("initializations_sha256") != initializations_sha256
        or record.get("implementation_sha256") != implementation_sha256
        or record.get("arrays") != file_record(arrays_path)
    ):
        return None
    arrays = load_npz(arrays_path)
    if set(arrays) != {
        "errors",
        "visible_hamming_changes",
        "full_state_hamming_changes",
    } or any(array.shape != (66, 257) for array in arrays.values()):
        return None
    return record["mode"], arrays


def save_profile_cache(
    cache_directory: Path,
    rule_sha256: str,
    config_sha256: str,
    initializations_sha256: str,
    implementation_sha256: str,
    mode: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    cache_directory.mkdir(parents=True, exist_ok=True)
    record_path, arrays_path = profile_cache_paths(cache_directory, rule_sha256)
    temporary = arrays_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(arrays_path)
    write_json(
        record_path,
        {
            "schema_version": 1,
            "rule_sha256": rule_sha256,
            "config_sha256": config_sha256,
            "initializations_sha256": initializations_sha256,
            "implementation_sha256": implementation_sha256,
            "mode": mode,
            "arrays": file_record(arrays_path),
        },
    )


def validate_sources(
    manifests: list[dict[str, Any]],
    artifact_root: Path,
    original_inputs: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    subset.base.validate_algebra()
    refine.validate_refinement()
    cycle_checks = validate_cycle_bookkeeping()
    records = []
    selected_errors: dict[tuple[str, str], list[int]] = {}
    selected_names = set(subset.NAMES)
    simple_seconds = None
    larger_seconds = None
    for manifest in manifests:
        checkpoint_ref = source_reference(manifest, "checkpoint_update_500.npz")
        trajectory_ref = source_reference(manifest, "probe_trajectory_update_500.npz")
        directory = artifact_root / manifest["run_id"]
        verify_file(directory / "checkpoint_update_500.npz", checkpoint_ref)
        verify_file(directory / "probe_trajectory_update_500.npz", trajectory_ref)
        saved = load_npz(directory / "probe_trajectory_update_500.npz")
        for hardening in ("common", "native"):
            mode_started = time.perf_counter()
            gates, wires, dag, identities = load_rule(manifest, hardening, artifact_root)
            simplified = subset.hard_rollout(original_inputs, dag, 20)
            errors = np.sum(simplified[-1, ..., 0] != target, axis=(1, 2))
            expected = manifest["evaluations"][-1][hardening]
            if int(errors.sum()) != expected["bit_errors"] or int(np.sum(errors == 0)) != expected["perfect_grid_count"]:
                raise ValueError(f"tick-20 metric mismatch: {manifest['run_id']}/{hardening}")
            full_first = subset.base.rollout(
                original_inputs[:1], lambda state: subset.base.full_step(state, gates, wires), 20
            )
            if not np.array_equal(full_first, simplified[:, :1]):
                raise ValueError(f"full/simplified mismatch: {manifest['run_id']}/{hardening}")
            if not np.array_equal(simplified[:, :1], saved[hardening]):
                raise ValueError(f"saved trajectory mismatch: {manifest['run_id']}/{hardening}")
            if manifest["run_id"] in selected_names:
                extended = subset.hard_rollout(original_inputs, dag, 64)
                selected_errors[(manifest["run_id"], hardening)] = np.sum(
                    extended[..., 0] != target, axis=(1, 2, 3)
                ).astype(int).tolist()
            elapsed = time.perf_counter() - mode_started
            if manifest["run_id"].endswith("seed04_categorical_reference_decay") and hardening == "common":
                simple_seconds = elapsed
            if manifest["run_id"].endswith("seed08_truth_reference_decay") and hardening == "common":
                larger_seconds = elapsed
            records.append({"run_id": manifest["run_id"], "hardening": hardening, **identities})

    for hardening in ("common", "native"):
        for run in (
            "R002_formal_attempt01_seed04_categorical_reference_decay",
            "R002_formal_attempt01_seed04_categorical_no_decay",
        ):
            assert selected_errors[(run, hardening)][20:] == [0] * 45
        assert selected_errors[("R002_formal_attempt01_seed13_categorical_reference_decay", hardening)][20:23] == [44, 2, 0]
        seed14 = selected_errors[("R002_formal_attempt01_seed14_truth_reference_decay", hardening)]
        assert seed14[20] == 0 and seed14[21] == 4096 and seed14[22] == 0
        assert selected_errors[("R002_formal_attempt01_seed08_truth_reference_decay", hardening)][20] == 2472

    certificate_expectations = {
        "R002_formal_attempt01_seed04_categorical_reference_decay": ("all_initial_states_visible_target_certified", 16),
        "R002_formal_attempt01_seed04_categorical_no_decay": ("all_initial_states_visible_target_certified", 16),
        "R002_formal_attempt01_seed13_categorical_reference_decay": ("all_initial_states_visible_target_certified", 25),
        "R002_formal_attempt01_seed15_categorical_no_decay": ("all_initial_states_visible_target_certified", 20),
        "R002_formal_attempt01_seed08_truth_reference_decay": ("persistent_visible_failure_certified", None),
    }
    certificates = {}
    for manifest in manifests:
        if manifest["run_id"] not in certificate_expectations:
            continue
        _, _, dag, _ = load_rule(manifest, "common", artifact_root)
        certificate = universal_certificate(dag, target, 256)
        status, tick = certificate_expectations[manifest["run_id"]]
        if certificate["status"] != status or certificate["certified_visible_settling_tick"] != tick:
            raise ValueError(f"certificate regression: {manifest['run_id']}")
        certificates[manifest["run_id"]] = certificate

    witness = json.loads((REVIEW_DIR / "seed14_phase_counterexample.json").read_text())
    packed = np.frombuffer(bytes.fromhex(witness["initial_state_packed_hex"]), dtype=np.uint8)
    initial = np.unpackbits(packed, bitorder="big").reshape(1, 16, 16, 8)
    seed14_manifest = next(
        item for item in manifests if item["run_id"] == witness["run_id"]
    )
    _, _, seed14_dag, _ = load_rule(seed14_manifest, "common", artifact_root)
    replay = subset.hard_rollout(initial, seed14_dag, 20)
    if int(np.sum(replay[-1, 0, ..., 0] != target)) != 128:
        raise ValueError("published seed-14 phase witness regression")
    return {
        "status": "passed",
        "source_modes_verified": len(records),
        "unique_source_artifacts_verified": len(manifests) * 4,
        "artifact_identity_check_operations": len(records) * 2 + len(manifests) * 2,
        "tick20_metrics_exact": True,
        "saved_first_probe_trajectories_exact": True,
        "unsimplified_vs_simplified_exact": True,
        "cycle_bookkeeping": cycle_checks,
        "selected_six_run_regressions_exact": True,
        "selected_certificates": certificates,
        "published_seed14_witness_exact": True,
        "profile_basis_seconds": {
            "simple_validation_mode": simple_seconds,
            "larger_validation_mode": larger_seconds,
        },
        "sources": records,
        "seconds": time.perf_counter() - started,
    }


def aggregate_results(modes: list[dict[str, Any]], report_ticks: list[int]) -> dict[str, Any]:
    table = []
    per_seed = []
    for condition in EXPECTED_CONDITIONS:
        for hardening in ("common", "native"):
            selected = [
                mode for mode in modes if mode["condition"] == condition and mode["hardening"] == hardening
            ]
            row = {
                "condition": condition,
                "hardening": hardening,
                "run_count": len(selected),
                "original_tick20_success_count": sum(
                    mode["initialization_groups"]["original_probe"]["report_ticks"]["20"]["all_correct"]
                    for mode in selected
                ),
                "original_all_correct_run_count_by_tick": {
                    str(tick): sum(
                        mode["initialization_groups"]["original_probe"]["report_ticks"][str(tick)]["all_correct"]
                        for mode in selected
                    )
                    for tick in report_ticks
                },
                "universally_eventually_correct_count": sum(
                    mode["universal_state_certificate"]["status"]
                    == "all_initial_states_visible_target_certified"
                    for mode in selected
                ),
                "mixed_cycle_run_count": sum(
                    any(record["cycle_behavior"] == "some_phases_correct" for record in mode["trajectories"])
                    for mode in selected
                ),
                "run_with_unresolved_trajectory_count": sum(
                    any(record["cycle_entry_tick"] is None for record in mode["trajectories"])
                    for mode in selected
                ),
                "certified_persistent_failure_count": sum(
                    mode["universal_state_certificate"]["status"]
                    == "persistent_visible_failure_certified"
                    for mode in selected
                ),
            }
            table.append(row)
    for seed in range(16):
        for condition in EXPECTED_CONDITIONS:
            for hardening in ("common", "native"):
                mode = next(
                    item
                    for item in modes
                    if item["seed"] == seed
                    and item["condition"] == condition
                    and item["hardening"] == hardening
                )
                per_seed.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "hardening": hardening,
                        "original": mode["initialization_groups"]["original_probe"]["report_ticks"],
                        "fresh": mode["initialization_groups"]["fresh_probe"]["report_ticks"],
                        "all_zero": mode["initialization_groups"]["all_zero"]["report_ticks"],
                        "all_one": mode["initialization_groups"]["all_one"]["report_ticks"],
                        "universal": compact_certificate(
                            mode["universal_state_certificate"]
                        ),
                    }
                )
    original_failures = [
        mode
        for mode in modes
        if not mode["initialization_groups"]["original_probe"]["report_ticks"]["20"]["all_correct"]
    ]
    late = [
        mode
        for mode in original_failures
        if mode["universal_state_certificate"]["status"]
        == "all_initial_states_visible_target_certified"
        or any(
            mode["initialization_groups"]["original_probe"]["report_ticks"][str(tick)]["all_correct"]
            for tick in report_ticks
            if tick > 20
        )
    ]
    original_successes = [mode for mode in modes if mode not in original_failures]
    dependent = [
        mode
        for mode in original_successes
        if mode["universal_state_certificate"]["status"]
        != "all_initial_states_visible_target_certified"
        or any(record["cycle_behavior"] == "some_phases_correct" for record in mode["trajectories"])
    ]
    persistent_failures = [
        mode
        for mode in modes
        if mode["universal_state_certificate"]["status"]
        == "persistent_visible_failure_certified"
    ]
    unresolved = [
        mode
        for mode in modes
        if any(record["cycle_entry_tick"] is None for record in mode["trajectories"])
        and mode["universal_state_certificate"]["status"] == "inconclusive"
    ]
    sampled_initialization_failures = [
        mode
        for mode in original_successes
        if any(
            not mode["initialization_groups"][group]["report_ticks"]["20"]["all_correct"]
            for group in ("fresh_probe", "all_zero", "all_one")
        )
    ]
    return {
        "main_table": table,
        "per_seed": per_seed,
        "headline": {
            "original_failure_mode_count": len(original_failures),
            "demonstrably_late_generator_mode_count": len(late),
            "demonstrably_late_generator_modes": [f"{m['run_id']}::{m['hardening']}" for m in late],
            "original_success_mode_count": len(original_successes),
            "phase_or_initialization_dependent_success_mode_count": len(dependent),
            "phase_or_initialization_dependent_success_modes": [f"{m['run_id']}::{m['hardening']}" for m in dependent],
            "certified_persistent_failure_mode_count": len(persistent_failures),
            "unresolved_mode_count": len(unresolved),
            "original_success_with_sampled_initialization_failure_mode_count": len(
                sampled_initialization_failures
            ),
            "primary_common_hardening": {
                "original_failure_count": sum(
                    mode["hardening"] == "common" for mode in original_failures
                ),
                "demonstrably_late_generator_count": sum(
                    mode["hardening"] == "common" for mode in late
                ),
                "original_success_count": sum(
                    mode["hardening"] == "common" for mode in original_successes
                ),
                "phase_or_initialization_dependent_success_count": sum(
                    mode["hardening"] == "common" for mode in dependent
                ),
                "certified_persistent_failure_count": sum(
                    mode["hardening"] == "common" for mode in persistent_failures
                ),
                "unresolved_count": sum(
                    mode["hardening"] == "common" for mode in unresolved
                ),
            },
        },
    }


def render_svg(path: Path, modes: list[dict[str, Any]], horizon: int) -> None:
    width, height = 920, 520
    left, right, top, bottom = 75, 25, 45, 65
    colors = {
        "categorical_reference_decay": "#0072B2",
        "truth_reference_decay": "#D55E00",
        "categorical_no_decay": "#56B4E9",
        "truth_no_decay": "#CC79A7",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="460" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">R002 common-hard original-probe errors after optimizer update 500</text>',
    ]
    selected = [mode for mode in modes if mode["hardening"] == "common"]
    maximum = max(
        max(mode["initialization_groups"]["original_probe"]["aggregate_errors_by_tick"])
        for mode in selected
    )
    for condition in EXPECTED_CONDITIONS:
        rows = [
            mode["initialization_groups"]["original_probe"]["aggregate_errors_by_tick"]
            for mode in selected
            if mode["condition"] == condition
        ]
        median = np.median(np.asarray(rows), axis=0)
        points = []
        for tick, value in enumerate(median):
            x = left + tick / horizon * (width - left - right)
            y = top + (1 - value / maximum) * (height - top - bottom)
            points.append(f"{x:.2f},{y:.2f}")
        lines.append(
            f'<polyline fill="none" stroke="{colors[condition]}" stroke-width="2" points="{" ".join(points)}"/>'
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
            f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif">runtime tick</text>',
            f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">median aggregate errors over 32 original probes</text>',
        ]
    )
    for index, condition in enumerate(EXPECTED_CONDITIONS):
        y = 55 + index * 20
        lines.append(f'<line x1="610" y1="{y}" x2="635" y2="{y}" stroke="{colors[condition]}" stroke-width="3"/>')
        lines.append(f'<text x="642" y="{y+4}" font-family="sans-serif" font-size="12">{condition}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def render_timing_svg(path: Path, modes: list[dict[str, Any]], horizon: int) -> None:
    width, height = 920, 500
    left, right, top, bottom = 70, 25, 55, 65
    bins = np.arange(0, horizon + 17, 16)
    certified = [
        mode["universal_state_certificate"]["certified_visible_settling_tick"]
        for mode in modes
        if mode["universal_state_certificate"]["status"]
        == "all_initial_states_visible_target_certified"
    ]
    observed = [
        mode["initialization_groups"]["original_probe"]["first_simultaneous_hit_tick"]
        for mode in modes
    ]
    observed_finite = [value for value in observed if value is not None]
    certified_counts, _ = np.histogram(certified, bins=bins)
    observed_counts, _ = np.histogram(observed_finite, bins=bins)
    maximum = max(1, int(max(certified_counts.max(), observed_counts.max())))
    plot_width = width - left - right
    plot_height = height - top - bottom
    bar_width = plot_width / len(certified_counts)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="460" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">R002 runtime timing after optimizer update 500</text>',
        f'<text x="460" y="44" text-anchor="middle" font-family="sans-serif" font-size="12">observed original-probe hits: n={len(observed_finite)}, unresolved/no hit: n={len(observed)-len(observed_finite)}; universal target certificates: n={len(certified)}</text>',
    ]
    for index, (observed_count, certified_count) in enumerate(
        zip(observed_counts, certified_counts)
    ):
        x = left + index * bar_width
        observed_height = observed_count / maximum * plot_height
        certified_height = certified_count / maximum * plot_height
        lines.append(
            f'<rect x="{x + 1:.2f}" y="{top + plot_height - observed_height:.2f}" width="{bar_width / 2 - 2:.2f}" height="{observed_height:.2f}" fill="#0072B2"/>'
        )
        lines.append(
            f'<rect x="{x + bar_width / 2:.2f}" y="{top + plot_height - certified_height:.2f}" width="{bar_width / 2 - 2:.2f}" height="{certified_height:.2f}" fill="#D55E00"/>'
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
            f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif">runtime tick (16-tick bins)</text>',
            f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">run/hardening modes</text>',
            '<rect x="650" y="65" width="14" height="10" fill="#0072B2"/><text x="670" y="75" font-family="sans-serif" font-size="12">first simultaneous original-probe hit</text>',
            '<rect x="650" y="83" width="14" height="10" fill="#D55E00"/><text x="670" y="93" font-family="sans-serif" font-size="12">universal certified onset</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def render_structure_svg(path: Path, modes: list[dict[str, Any]], horizon: int) -> None:
    width, height = 920, 520
    left, right, top, bottom = 75, 25, 55, 65
    colors = {
        "categorical_reference_decay": "#0072B2",
        "truth_reference_decay": "#D55E00",
        "categorical_no_decay": "#56B4E9",
        "truth_no_decay": "#CC79A7",
    }
    selected = [mode for mode in modes if mode["hardening"] == "common"]
    finite = [
        mode
        for mode in selected
        if mode["initialization_groups"]["original_probe"]["first_simultaneous_hit_tick"]
        is not None
    ]
    max_nodes = max(mode["structure"]["visible_core_binary_nodes"] for mode in selected)
    plot_width = width - left - right
    plot_height = height - top - bottom
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="460" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">Visible-core size vs first simultaneous original-probe hit</text>',
        f'<text x="460" y="44" text-anchor="middle" font-family="sans-serif" font-size="12">common hardening; finite hits n={len(finite)}, no hit through tick {horizon}: n={len(selected)-len(finite)}</text>',
    ]
    for mode in finite:
        nodes = mode["structure"]["visible_core_binary_nodes"]
        tick = mode["initialization_groups"]["original_probe"]["first_simultaneous_hit_tick"]
        x = left + nodes / max(1, max_nodes) * plot_width
        y = top + tick / horizon * plot_height
        lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{colors[mode["condition"]]}" fill-opacity="0.75"/>'
        )
    for index, condition in enumerate(EXPECTED_CONDITIONS):
        y = 68 + index * 18
        lines.append(
            f'<circle cx="650" cy="{y}" r="4" fill="{colors[condition]}"/>'
        )
        lines.append(
            f'<text x="660" y="{y+4}" font-family="sans-serif" font-size="11">{condition}</text>'
        )
    lines.extend(
        [
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
            f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif">visible recurrent-core binary operations</text>',
            f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif">first simultaneous hit runtime tick</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def summary_markdown(analysis_id: str, aggregate: dict[str, Any]) -> str:
    headline = aggregate["headline"]
    primary = headline["primary_common_hardening"]
    rows = []
    for row in aggregate["main_table"]:
        later = row["original_all_correct_run_count_by_tick"]
        rows.append(
            f"| {row['condition']} | {row['hardening']} | {row['original_tick20_success_count']} | "
            + " / ".join(str(later[str(tick)]) for tick in (32, 64, 128, 256))
            + f" | {row['universally_eventually_correct_count']} | {row['mixed_cycle_run_count']} | {row['run_with_unresolved_trajectory_count']} |"
        )
    return f"""# R002 full-cohort runtime profile

Analysis ID: `{analysis_id}`. Status: completed. This is a post-hoc execution of
the frozen update-500 Boolean circuits through runtime tick 256; it does not
change the original tick-20 experiment outcomes.

Across the 128 run/hardening modes, {headline['demonstrably_late_generator_mode_count']}
original tick-20 failures are demonstrably late generators under at least one
declared later readout or a universal-state certificate. The primary common-hard
count is {primary['demonstrably_late_generator_count']} of
{primary['original_failure_count']} circuits.
{headline['phase_or_initialization_dependent_success_mode_count']} original-success
modes depend on phase or lack an initialization-independent certificate (the
primary common-hard count is {primary['phase_or_initialization_dependent_success_count']}
of {primary['original_success_count']}).
{headline['original_success_with_sampled_initialization_failure_mode_count']}
original-success modes fail at tick 20 on at least one sampled fresh, all-zero,
or all-one initialization group.
{headline['certified_persistent_failure_mode_count']} modes have a certified
persistent visible failure, while {headline['unresolved_mode_count']} modes remain
unresolved by the replay and abstraction cap. The corresponding primary
common-hard counts are {primary['certified_persistent_failure_count']} and
{primary['unresolved_count']} circuits.

## Main comparison

Later counts are the number of circuits correct simultaneously on all 32
original probes at ticks 32 / 64 / 128 / 256. Mixed-cycle and unresolved counts
mean at least one of the 66 ordinary initializations for that run/hardening mode.
These flags may overlap.

| Condition | Hardening | Original success at 20 | Original all-correct at 32 / 64 / 128 / 256 | Universal eventual target | Any mixed cycle | Any unresolved orbit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Fresh probes and all-zero/all-one initializations are retained separately in
`aggregate.json` and `runs.json`. `trajectories.jsonl.gz` contains one compact
record per run, hardening, and initialization, and `per_seed.csv` keeps the
paired design directly inspectable. Every-tick error, perfect-count, certificate,
and Hamming
arrays are stored outside git in the hashed metrics artifact named by the
manifest. Phase witnesses are likewise external artifacts.

![Median runtime error trajectories](runtime_errors.svg)

![Observed and certified runtime timing](runtime_timing.svg)

![Circuit structure against runtime behavior](structure_vs_runtime.svg)

The fixed 32 probes, fresh 32 probes, two constant states, runtime ticks, and
two hardening exports are not independent training trials. A correct suffix at
tick 256 is finite replay evidence unless a full-state cycle or universal-state
certificate establishes persistence. Circuit counts are constructive
simplifications, not minimum descriptions or Kolmogorov complexity.
"""


def main() -> None:
    analysis_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--backend", choices=("numpy",), default="numpy")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    artifact_root = args.artifact_root.resolve()
    manifests = enumerate_cohort(artifact_root)
    result_directory = ROOT / "results" / args.analysis_id
    output_artifact_root = ROOT / "artifacts" / args.analysis_id
    existing_manifest_path = result_directory / "manifest.json"
    existing = None
    if existing_manifest_path.exists():
        existing = json.loads(existing_manifest_path.read_text())
        if existing.get("analysis_id") != args.analysis_id or existing.get("config_sha256") != config_sha256:
            raise ValueError("existing analysis directory is incompatible")
        if existing.get("status") == "completed" and not args.validate_only:
            raise ValueError("analysis is already completed")
    provenance = existing["analysis_code"] if existing else code_provenance()
    result_directory.mkdir(parents=True, exist_ok=True)
    output_artifact_root.mkdir(parents=True, exist_ok=True)

    original_path = ROOT / "artifacts/R001_20260909T231016Z_seed23_jaxgpu_attempt01/fixed_evaluation_set.npz"
    probe_expected = manifests[0]["probe"]
    original_identity = verify_file(original_path, probe_expected)
    if original_identity["sha256"] != config["original_probe_sha256"]:
        raise ValueError("original probe hash does not match protocol")
    original = load_npz(original_path)
    original_inputs = original["inputs"].astype(np.uint8)
    target = original["target"][0, ..., 0].astype(np.uint8)
    if original_inputs.shape != (32, 16, 16, 8):
        raise ValueError("unexpected original probe shape")

    fresh_path = output_artifact_root / "fresh_probe.npz"
    generator = np.random.Generator(np.random.PCG64(20260910))
    fresh = generator.integers(0, 2, size=(32, 16, 16, 8), dtype=np.uint8)
    if fresh_path.exists():
        saved_fresh = load_npz(fresh_path)["inputs"]
        if not np.array_equal(saved_fresh, fresh):
            raise ValueError("existing fresh probe is incompatible")
    else:
        np.savez_compressed(fresh_path, inputs=fresh)
    fresh_identity = {
        **file_record(fresh_path),
        "shape": list(fresh.shape),
        "dtype": str(fresh.dtype),
        "raw_array_sha256": sha256_array(fresh),
        "generator": "numpy.random.Generator(numpy.random.PCG64(20260910))",
        "numpy_version": np.__version__,
    }
    initializations, labels = make_initializations(original_inputs, fresh)

    manifest = {
        "schema_version": 1,
        "analysis_id": args.analysis_id,
        "protocol_id": config["protocol_id"],
        "status": "validating",
        "config_path": args.config.as_posix(),
        "config_sha256": config_sha256,
        "config": config,
        "source_training_commit": config["training_code_commit"],
        "analysis_code": provenance,
        "backend": args.backend,
        "workers": args.workers,
        "execution": {
            "device": "CPU",
            "hardware": {
                "cpu_model": cpu_model_name(),
                "machine": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
            },
            "compilation": "none",
            "executor": "independent NumPy Boolean/integer executor",
            "simplifier": config["simplifier"],
            "worker_selection": (
                "one process selected after timing one simple and one larger validation mode; "
                "full projected cost was under one minute and avoided process startup/oversubscription"
            ),
            "preflight_full_profile_seconds": {
                "seed04_categorical_reference_decay_common_visible_core_2": 0.2345,
                "seed08_truth_reference_decay_common_visible_core_145": 0.7348,
            },
        },
        "thread_settings": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
            "lock": file_record(ROOT / "envs/jax-gpu/uv.lock"),
        },
        "original_probe": original_identity,
        "fresh_probe": fresh_identity,
        "source_run_count": len(manifests),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "failures": [],
    }
    write_json(existing_manifest_path, manifest)
    try:
        validation = validate_sources(manifests, artifact_root, original_inputs, target)
        write_json(result_directory / "validation.json", validation)
        manifest["validation"] = {
            "status": "passed",
            "path": "validation.json",
            "sha256": sha256_file(result_directory / "validation.json"),
        }
        manifest["status"] = "validated" if args.validate_only else "running"
        write_json(existing_manifest_path, manifest)
        if args.validate_only:
            print(json.dumps({"status": "validated", "validation": validation}, indent=2))
            return

        source_lookup = {
            (record["run_id"], record["hardening"]): record
            for record in validation["sources"]
        }
        logical_modes = [
            (manifest_source, hardening, source_lookup[(manifest_source["run_id"], hardening)])
            for manifest_source in manifests
            for hardening in config["hardening_modes"]
        ]
        representatives: dict[str, tuple[dict[str, Any], str, dict[str, Any]]] = {}
        for logical_mode in logical_modes:
            representatives.setdefault(logical_mode[2]["rule_sha256"], logical_mode)

        cache_directory = output_artifact_root / "resume"
        initializations_sha256 = sha256_array(initializations)
        implementation_sha256 = sha256_file(Path(__file__))
        profiles: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
        missing: list[tuple[str, tuple[Any, ...]]] = []
        for rule_sha256, (manifest_source, hardening, _) in representatives.items():
            cached = load_profile_cache(
                cache_directory,
                rule_sha256,
                config_sha256,
                initializations_sha256,
                implementation_sha256,
            )
            if cached is not None:
                profiles[rule_sha256] = cached
                print(f"RESUME {manifest_source['run_id']} {hardening}", flush=True)
                continue
            arguments = (
                manifest_source,
                hardening,
                artifact_root,
                initializations,
                labels,
                target,
                config,
                output_artifact_root,
            )
            missing.append((rule_sha256, arguments))

        if args.workers == 1:
            for rule_sha256, arguments in missing:
                print(f"PROFILE {arguments[0]['run_id']} {arguments[1]}", flush=True)
                result = profile_mode_task(arguments)
                profiles[rule_sha256] = result
                save_profile_cache(
                    cache_directory,
                    rule_sha256,
                    config_sha256,
                    initializations_sha256,
                    implementation_sha256,
                    *result,
                )
        elif missing:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=context
            ) as executor:
                futures = {
                    executor.submit(profile_mode_task, arguments): (rule_sha256, arguments)
                    for rule_sha256, arguments in missing
                }
                for future in concurrent.futures.as_completed(futures):
                    rule_sha256, arguments = futures[future]
                    print(f"PROFILE {arguments[0]['run_id']} {arguments[1]}", flush=True)
                    result = future.result()
                    profiles[rule_sha256] = result
                    save_profile_cache(
                        cache_directory,
                        rule_sha256,
                        config_sha256,
                        initializations_sha256,
                        implementation_sha256,
                        *result,
                    )

        modes = []
        metric_arrays = []
        mode_labels = []
        for manifest_source, hardening, source in logical_modes:
            rule_sha256 = source["rule_sha256"]
            representative_source, representative_hardening, _ = representatives[
                rule_sha256
            ]
            base_mode, arrays = profiles[rule_sha256]
            mode = copy.deepcopy(base_mode)
            reused = (
                representative_source["run_id"] != manifest_source["run_id"]
                or representative_hardening != hardening
            )
            mode.update(
                {
                    "run_id": manifest_source["run_id"],
                    "seed": manifest_source["seed"],
                    "condition": manifest_source["condition"],
                    "representation": manifest_source["representation"],
                    "hardening": hardening,
                    "source": {
                        "export": source["export"],
                        "checkpoint": source["checkpoint"],
                        "rule_sha256": rule_sha256,
                    },
                    "profile_reused": reused,
                    "reused_from": (
                        f"{representative_source['run_id']}::{representative_hardening}"
                        if reused
                        else None
                    ),
                    "profile_seconds": 0.0 if reused else base_mode["profile_seconds"],
                }
            )
            phase_witness = mode.pop("_phase_witness", None)
            if phase_witness is not None:
                witness_path = (
                    output_artifact_root
                    / "witnesses"
                    / f"{manifest_source['run_id']}__{hardening}.npz"
                )
                mode["phase_witnesses"] = [
                    write_phase_witness(
                        witness_path,
                        phase_witness,
                        manifest_source["run_id"],
                        hardening,
                    )
                ]
            modes.append(mode)
            metric_arrays.append(arrays)
            mode_labels.append(f"{mode['run_id']}::{hardening}")

        metrics_path = output_artifact_root / "tick_metrics.npz"
        group_aggregate_errors = np.asarray(
            [
                [
                    mode["initialization_groups"][group]["aggregate_errors_by_tick"]
                    for group in INITIALIZATION_GROUPS
                ]
                for mode in modes
            ],
            dtype=np.uint16,
        )
        group_perfect_initializations = np.asarray(
            [
                [
                    mode["initialization_groups"][group]["perfect_initializations_by_tick"]
                    for group in INITIALIZATION_GROUPS
                ]
                for mode in modes
            ],
            dtype=np.uint8,
        )
        certificate_counts = np.full((len(modes), 257, 3), -1, dtype=np.int16)
        for mode_index, mode in enumerate(modes):
            for count in mode["universal_state_certificate"]["counts_by_tick"]:
                certificate_counts[mode_index, count["tick"]] = (
                    count["known_correct"],
                    count["known_wrong"],
                    count["unknown"],
                )
        np.savez_compressed(
            metrics_path,
            mode_labels=np.asarray(mode_labels),
            initialization_ids=np.asarray([item["initialization_id"] for item in labels]),
            initialization_groups=np.asarray(INITIALIZATION_GROUPS),
            errors=np.stack([item["errors"] for item in metric_arrays]),
            visible_hamming_changes=np.stack(
                [item["visible_hamming_changes"] for item in metric_arrays]
            ),
            full_state_hamming_changes=np.stack(
                [item["full_state_hamming_changes"] for item in metric_arrays]
            ),
            group_aggregate_errors=group_aggregate_errors,
            group_perfect_initializations=group_perfect_initializations,
            certificate_counts_known_correct_wrong_unknown=certificate_counts,
        )
        aggregate = aggregate_results(modes, config["report_ticks"])
        write_json(result_directory / "aggregate.json", aggregate)
        with (result_directory / "runs.json").open("w") as handle:
            compact_modes = []
            for mode in modes:
                compact = {
                    key: value
                    for key, value in mode.items()
                    if key not in {"trajectories", "universal_state_certificate", "initialization_groups", "structure"}
                }
                compact["universal_state_certificate"] = compact_certificate(
                    mode["universal_state_certificate"]
                )
                compact["initialization_groups"] = {
                    group: compact_group_summary(summary)
                    for group, summary in mode["initialization_groups"].items()
                }
                compact["structure"] = compact_structure(mode["structure"])
                compact_modes.append(compact)
            json.dump({"modes": compact_modes}, handle, indent=2)
            handle.write("\n")
        trajectories_path = result_directory / "trajectories.jsonl.gz"
        with trajectories_path.open("wb") as raw_handle:
            handle = gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0)
            for mode in modes:
                for record in mode["trajectories"]:
                    line = json.dumps(
                        {
                            "run_id": mode["run_id"],
                            "condition": mode["condition"],
                            "hardening": mode["hardening"],
                            **record,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    handle.write((line + "\n").encode())
            handle.close()
        per_seed_path = result_directory / "per_seed.csv"
        with per_seed_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "seed",
                    "condition",
                    "hardening",
                    "original_all_correct_ticks",
                    "fresh_all_correct_ticks",
                    "all_zero_correct_ticks",
                    "all_one_correct_ticks",
                    "universal_status",
                    "certified_visible_settling_tick",
                    "mixed_cycle_initialization_count",
                    "unresolved_trajectory_count",
                    "visible_core_binary_nodes",
                ]
            )
            for mode in modes:
                writer.writerow(
                    [
                        mode["seed"],
                        mode["condition"],
                        mode["hardening"],
                        ";".join(
                            tick
                            for tick, value in mode["initialization_groups"]["original_probe"]["report_ticks"].items()
                            if value["all_correct"]
                        ),
                        ";".join(
                            tick
                            for tick, value in mode["initialization_groups"]["fresh_probe"]["report_ticks"].items()
                            if value["all_correct"]
                        ),
                        ";".join(
                            tick
                            for tick, value in mode["initialization_groups"]["all_zero"]["report_ticks"].items()
                            if value["all_correct"]
                        ),
                        ";".join(
                            tick
                            for tick, value in mode["initialization_groups"]["all_one"]["report_ticks"].items()
                            if value["all_correct"]
                        ),
                        mode["universal_state_certificate"]["status"],
                        mode["universal_state_certificate"]["certified_visible_settling_tick"],
                        sum(
                            record["cycle_behavior"] == "some_phases_correct"
                            for record in mode["trajectories"]
                        ),
                        sum(
                            record["cycle_entry_tick"] is None
                            for record in mode["trajectories"]
                        ),
                        mode["structure"]["visible_core_binary_nodes"],
                    ]
                )
        render_svg(result_directory / "runtime_errors.svg", modes, config["horizon"])
        render_timing_svg(result_directory / "runtime_timing.svg", modes, config["horizon"])
        render_structure_svg(
            result_directory / "structure_vs_runtime.svg", modes, config["horizon"]
        )
        (result_directory / "summary.md").write_text(summary_markdown(args.analysis_id, aggregate))
        witnesses = sorted((output_artifact_root / "witnesses").glob("*.npz"))
        result_paths = (
            result_directory / "validation.json",
            result_directory / "aggregate.json",
            result_directory / "runs.json",
            trajectories_path,
            per_seed_path,
            result_directory / "runtime_errors.svg",
            result_directory / "runtime_timing.svg",
            result_directory / "structure_vs_runtime.svg",
            result_directory / "summary.md",
        )
        bundle_path = output_artifact_root / f"{args.analysis_id}_review.tar.gz"
        review_bundle = build_bundle(
            bundle_path,
            "R002",
            args.analysis_id,
            "runtime_profile_review",
            [*result_paths, fresh_path, metrics_path, *witnesses],
        )
        review_bundle["artifact"] = file_record(bundle_path)
        manifest.update(
            {
                "status": "completed",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "mode_count": len(modes),
                "trajectory_count": len(modes) * len(labels),
                "unique_rule_count": len(representatives),
                "profile_reuse_count": len(modes) - len(representatives),
                "timings_seconds": {
                    "validation": validation["seconds"],
                    "recorded_unique_rule_profile_cpu": sum(
                        mode["profile_seconds"] for mode, _ in profiles.values()
                    ),
                    "profiles_executed_this_invocation": len(missing),
                    "profiles_restored_from_cache": len(representatives) - len(missing),
                    "total_wall": time.perf_counter() - analysis_started,
                    "profile_basis": validation["profile_basis_seconds"],
                },
                "resume_cache": {
                    "path": cache_directory.relative_to(ROOT).as_posix(),
                    "completed_rule_records": len(
                        list(cache_directory.glob("*.json"))
                    ),
                    "completed_metric_records": len(
                        list(cache_directory.glob("*.npz"))
                    ),
                    "integrity_guard": (
                        "rule, protocol configuration, initialization bytes, "
                        "implementation, and compressed-array artifact hashes"
                    ),
                },
                "result_files": [file_record(path) for path in result_paths],
                "artifacts": [file_record(fresh_path), file_record(metrics_path)]
                + [file_record(path) for path in witnesses]
                + [file_record(bundle_path)],
                "review_bundle": review_bundle,
                "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            }
        )
        write_json(existing_manifest_path, manifest)
        print(json.dumps({"status": "completed", "headline": aggregate["headline"]}, indent=2))
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failures"].append({"type": type(error).__name__, "message": str(error)})
        write_json(existing_manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
