#!/usr/bin/env python3
"""Build the fixed compact R004 review subset and its identity manifest."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from recurrent_circuit_learning.r002 import CONDITIONS


SELECTED_UPDATES = (500, 1000, 1500, 2000)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, record: dict[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != record["bytes"]
        or sha256_file(path) != record["sha256"]
    ):
        raise ValueError(f"R004 review source identity mismatch: {path}")


def _artifact_path(root: Path, record: dict[str, Any], run_id: str) -> Path:
    path = Path(record["path"])
    if not path.is_file():
        path = root / "artifacts" / run_id / path.name
    _verify(path, record)
    return path


def _add(members: dict[str, Path], name: str, path: Path) -> None:
    if name in members:
        raise ValueError(f"duplicate R004 review member: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    members[name] = path


def collect_members(
    root: Path, sweep_id: str, analysis_id: str, preflight_id: str
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Collect and validate the fixed review selection."""
    members: dict[str, Path] = {}
    run_manifests = []
    for seed in range(16):
        for condition in CONDITIONS:
            run_id = f"R004_{sweep_id}_seed{seed:02d}_{condition}"
            result_directory = root / "results" / run_id
            manifest_path = result_directory / "manifest.json"
            metrics_path = result_directory / "metrics.jsonl"
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("run_id") != run_id
                or manifest.get("status") != "completed"
                or manifest.get("current_update") != 2000
            ):
                raise ValueError(f"incomplete R004 review source: {manifest_path}")
            _verify(metrics_path, manifest["metrics"])
            _add(members, f"results/{run_id}/manifest.json", manifest_path)
            _add(members, f"results/{run_id}/metrics.jsonl", metrics_path)
            run_manifests.append(manifest)

    aggregate_directory = root / "results" / f"R004_{sweep_id}"
    for path in sorted(aggregate_directory.iterdir()):
        # The outer release manifest records the review archive, so embedding it
        # here would create a circular identity dependency.
        if path.is_file() and path.name != "release-assets.json":
            _add(members, f"results/{aggregate_directory.name}/{path.name}", path)

    analysis_directory = root / "results" / f"R004_{sweep_id}_{analysis_id}"
    analysis_manifest = json.loads((analysis_directory / "manifest.json").read_text())
    if analysis_manifest.get("status") != "completed":
        raise ValueError("R004 review requires a completed analysis")
    for record in analysis_manifest["artifacts"]:
        path = analysis_directory / Path(record["path"]).name
        _verify(path, record)
    for path in sorted(analysis_directory.iterdir()):
        if path.is_file():
            _add(members, f"results/{analysis_directory.name}/{path.name}", path)

    preflight_directory = root / "results" / f"R004_{preflight_id}"
    validation = json.loads((preflight_directory / "validation.json").read_text())
    if validation.get("status") != "passed":
        raise ValueError("R004 review requires a passing preflight")
    for path in sorted(preflight_directory.iterdir()):
        if path.is_file():
            _add(members, f"results/{preflight_directory.name}/{path.name}", path)

    for relative in (
        "configs/experiments/r004_training_continuation_v1.json",
        "experiments/R004/HANDOFF.md",
        "experiments/R004/SPEC.md",
        "recurrent_circuit_learning/r004_runner.py",
        "recurrent_circuit_learning/r004_analysis.py",
        "scripts/r004.py",
        "scripts/build_r004_review.py",
        "scripts/build_release_assets.py",
        "tests/test_r004.py",
        "tests/test_release_assets.py",
        "envs/jax-gpu/pyproject.toml",
        "envs/jax-gpu/uv.lock",
    ):
        _add(members, f"implementation/{relative}", root / relative)

    with gzip.open(analysis_directory / "runtime_profiles.jsonl.gz", "rt") as handle:
        runtime_rows = [json.loads(line) for line in handle if line.strip()]
    selected_runtime_arrays: dict[str, dict[str, Any]] = {}
    for row in runtime_rows:
        if row["seed"] == 0 and row["global_update"] in SELECTED_UPDATES:
            selected_runtime_arrays[row["arrays"]["sha256"]] = row["arrays"]
    for digest, record in sorted(selected_runtime_arrays.items()):
        path = _artifact_path(root, record, f"R004_{sweep_id}_{analysis_id}")
        _add(members, f"derived/runtime_rules/{digest}.npz", path)

    for manifest in run_manifests:
        if manifest["seed"] != 0:
            continue
        run_id = manifest["run_id"]
        parent_run_id = manifest["lineage"]["parent_run_id"]
        parent_manifest_path = Path(manifest["lineage"]["parent_manifest"]["path"])
        _verify(parent_manifest_path, manifest["lineage"]["parent_manifest"])
        parent_manifest = json.loads(parent_manifest_path.read_text())
        parent_checkpoint = next(
            item for item in parent_manifest["checkpoints"] if item["update"] == 500
        )
        for label in ("checkpoint", "probe_trajectory"):
            record = parent_checkpoint[label]
            path = _artifact_path(root, record, parent_run_id)
            _add(members, f"source/{parent_run_id}/{path.name}", path)
        for record in parent_manifest["artifacts"]:
            path = _artifact_path(root, record, parent_run_id)
            _add(members, f"source/{parent_run_id}/{path.name}", path)

        for update in SELECTED_UPDATES[1:]:
            checkpoint = next(
                item for item in manifest["checkpoints"] if item["update"] == update
            )
            for label in ("checkpoint", "probe_trajectory"):
                record = checkpoint[label]
                path = _artifact_path(root, record, run_id)
                _add(members, f"continued/{run_id}/{path.name}", path)
            for export in manifest["structural_exports"]:
                if export["global_update"] != update:
                    continue
                record = export["artifact"]
                path = _artifact_path(root, record, run_id)
                _add(members, f"continued/{run_id}/{path.name}", path)

    references = {
        "schema_version": 1,
        "experiment_id": "R004",
        "sweep_id": sweep_id,
        "analysis_id": analysis_id,
        "selection_rule": {
            "seed": 0,
            "conditions": list(CONDITIONS),
            "global_updates": list(SELECTED_UPDATES),
            "hardening_modes": ["native", "common"],
        },
        "parent_release": {
            "tag": "experiment/R002",
            "url": "https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R002",
            "note": "Only the fixed seed-0 update-500 source subset is included; the complete R002 archive remains referenced by release tag.",
        },
        "included_completed_runs": len(run_manifests),
        "included_runtime_rule_arrays": len(selected_runtime_arrays),
    }
    return members, references


def _normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def build_review_archive(
    output_path: Path, members: dict[str, Path], references: dict[str, Any]
) -> dict[str, Any]:
    contents = [
        {
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in sorted(members.items())
    ]
    embedded = {
        **references,
        "kind": "R004_compact_review_subset",
        "files": contents,
    }
    embedded_bytes = (json.dumps(embedded, indent=2, sort_keys=True) + "\n").encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        output_path.open("wb") as raw,
        gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, mtime=0, fileobj=raw
        ) as zipped,
        tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name, payload in (
            ("CONTENTS.json", embedded_bytes),
            (
                "SOURCE_REFERENCES.json",
                (json.dumps(references, indent=2, sort_keys=True) + "\n").encode(),
            ),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(_normalize(info), io.BytesIO(payload))
        for name, path in sorted(members.items()):
            with path.open("rb") as handle:
                archive.addfile(
                    _normalize(archive.gettarinfo(str(path), arcname=name)), handle
                )
    return {
        "schema_version": 1,
        "experiment_id": "R004",
        "sweep_id": references["sweep_id"],
        "analysis_id": references["analysis_id"],
        "release_tag": "experiment/R004",
        "review_archive": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "file_count": len(contents),
        },
        "selection_rule": references["selection_rule"],
        "parent_release": references["parent_release"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--preflight-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    members, references = collect_members(
        root, args.sweep_id, args.analysis_id, args.preflight_id
    )
    result = build_review_archive(args.output.resolve(), members, references)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
