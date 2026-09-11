#!/usr/bin/env python3
"""Build deterministic GitHub release bundles for experiment artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any

EXPERIMENT_ID_PATTERN = re.compile(r"R[0-9]{3}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_record(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _normalized_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o644
    return info


def build_bundle(
    output_path: Path,
    experiment_id: str,
    run_id: str,
    bundle_kind: str,
    files: list[Path],
) -> dict[str, Any]:
    contents = [_content_record(path) for path in sorted(files)]
    embedded_manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "bundle_kind": bundle_kind,
        "files": contents,
    }
    manifest_bytes = (
        json.dumps(embedded_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()

    tar_buffer = io.BytesIO()
    with tarfile.open(
        fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT
    ) as archive:
        manifest_info = tarfile.TarInfo("CONTENTS.json")
        manifest_info.size = len(manifest_bytes)
        archive.addfile(_normalized_tar_info(manifest_info), io.BytesIO(manifest_bytes))
        for path in sorted(files):
            with path.open("rb") as handle:
                archive.addfile(
                    _normalized_tar_info(
                        archive.gettarinfo(str(path), arcname=path.name)
                    ),
                    handle,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        output_path.open("wb") as output_handle,
        gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, mtime=0, fileobj=output_handle
        ) as compressed,
    ):
        compressed.write(tar_buffer.getvalue())

    return {
        "name": output_path.name,
        "kind": bundle_kind,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "contents": contents,
    }


def build_tree_bundle(
    output_path: Path,
    experiment_id: str,
    sweep_id: str,
    bundle_kind: str,
    files: list[Path],
    artifact_root: Path,
) -> dict[str, Any]:
    contents = []
    for path in sorted(files):
        record = _content_record(path)
        record["path"] = path.relative_to(artifact_root).as_posix()
        del record["name"]
        contents.append(record)
    embedded_manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "sweep_id": sweep_id,
        "bundle_kind": bundle_kind,
        "files": contents,
    }
    manifest_bytes = (
        json.dumps(embedded_manifest, indent=2, sort_keys=True) + "\n"
    ).encode()

    tar_buffer = io.BytesIO()
    with tarfile.open(
        fileobj=tar_buffer, mode="w", format=tarfile.PAX_FORMAT
    ) as archive:
        manifest_info = tarfile.TarInfo("CONTENTS.json")
        manifest_info.size = len(manifest_bytes)
        archive.addfile(_normalized_tar_info(manifest_info), io.BytesIO(manifest_bytes))
        for path in sorted(files):
            with path.open("rb") as handle:
                archive.addfile(
                    _normalized_tar_info(
                        archive.gettarinfo(
                            str(path),
                            arcname=path.relative_to(artifact_root).as_posix(),
                        )
                    ),
                    handle,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        output_path.open("wb") as output_handle,
        gzip.GzipFile(
            filename="", mode="wb", compresslevel=9, mtime=0, fileobj=output_handle
        ) as compressed,
    ):
        compressed.write(tar_buffer.getvalue())

    return {
        "name": output_path.name,
        "kind": bundle_kind,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "contents": contents,
    }


def build_release_assets(
    experiment_id: str,
    run_id: str,
    artifact_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("experiment ID must have the form R000")
    if not run_id.startswith(f"{experiment_id}_"):
        raise ValueError("run ID must begin with the experiment ID")
    if not artifact_directory.is_dir():
        raise FileNotFoundError(artifact_directory)

    checkpoints = sorted(artifact_directory.glob("checkpoint_update_*.npz"))
    analysis_files = sorted(artifact_directory.glob("probe_trajectory_update_*.npz"))
    analysis_files.extend(
        artifact_directory / name
        for name in ("fixed_evaluation_set.npz", "final_hard_circuit.npz")
    )
    if not checkpoints or any(not path.is_file() for path in analysis_files):
        raise ValueError("artifact directory is missing checkpoint or analysis files")

    output_directory.mkdir(parents=True, exist_ok=True)
    assets = [
        build_bundle(
            output_directory / f"{run_id}_checkpoints.tar.gz",
            experiment_id,
            run_id,
            "checkpoints",
            checkpoints,
        ),
        build_bundle(
            output_directory / f"{run_id}_analysis.tar.gz",
            experiment_id,
            run_id,
            "analysis",
            analysis_files,
        ),
    ]
    release_manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "release_tag": f"experiment/{experiment_id}",
        "assets": assets,
    }
    manifest_path = output_directory / f"{run_id}_release-assets.json"
    manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n"
    )
    release_manifest["manifest_asset"] = {
        "name": manifest_path.name,
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    return release_manifest


def build_sweep_release_assets(
    experiment_id: str,
    sweep_id: str,
    artifact_root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("experiment ID must have the form R000")
    if not artifact_root.is_dir():
        raise FileNotFoundError(artifact_root)

    run_directories = sorted(artifact_root.glob(f"{experiment_id}_{sweep_id}_seed*"))
    if not run_directories or any(not path.is_dir() for path in run_directories):
        raise ValueError("artifact root has no matching sweep run directories")

    checkpoints: list[Path] = []
    analysis_files: list[Path] = []
    for run_directory in run_directories:
        run_checkpoints = sorted(run_directory.glob("checkpoint_update_*.npz"))
        run_trajectories = sorted(run_directory.glob("probe_trajectory_update_*.npz"))
        run_exports = [
            run_directory / "final_native_circuit.npz",
            run_directory / "final_common_circuit.npz",
        ]
        if (
            not run_checkpoints
            or len(run_checkpoints) != len(run_trajectories)
            or any(not path.is_file() for path in run_exports)
        ):
            raise ValueError(f"incomplete sweep artifact directory: {run_directory}")
        checkpoints.extend(run_checkpoints)
        analysis_files.extend(run_trajectories)
        analysis_files.extend(run_exports)

    output_directory.mkdir(parents=True, exist_ok=True)
    asset_prefix = f"{experiment_id}_{sweep_id}"
    assets = [
        build_tree_bundle(
            output_directory / f"{asset_prefix}_checkpoints.tar.gz",
            experiment_id,
            sweep_id,
            "checkpoints",
            checkpoints,
            artifact_root,
        ),
        build_tree_bundle(
            output_directory / f"{asset_prefix}_analysis.tar.gz",
            experiment_id,
            sweep_id,
            "analysis",
            analysis_files,
            artifact_root,
        ),
    ]
    release_manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "sweep_id": sweep_id,
        "release_tag": f"experiment/{experiment_id}",
        "run_directories": [path.name for path in run_directories],
        "assets": assets,
    }
    manifest_path = output_directory / f"{asset_prefix}_release-assets.json"
    manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n"
    )
    release_manifest["manifest_asset"] = {
        "name": manifest_path.name,
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    return release_manifest


def build_diagnostic_release_assets(
    experiment_id: str,
    run_id: str,
    artifact_directory: Path,
    review_archive: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Build a derived-array bundle plus an existing compact review archive."""
    if not EXPERIMENT_ID_PATTERN.fullmatch(experiment_id):
        raise ValueError("experiment ID must have the form R000")
    if not run_id.startswith(f"{experiment_id}_"):
        raise ValueError("run ID must begin with the experiment ID")
    if not artifact_directory.is_dir() or not review_archive.is_file():
        raise FileNotFoundError("diagnostic artifact directory or review archive")
    case_arrays = sorted(artifact_directory.glob("case_*.npz"))
    directions = artifact_directory / "state_jvp_directions.npy"
    if len(case_arrays) != 96 or not directions.is_file():
        raise ValueError("diagnostic artifact directory is incomplete")

    output_directory.mkdir(parents=True, exist_ok=True)
    derived = build_bundle(
        output_directory / f"{run_id}_derived.tar.gz",
        experiment_id,
        run_id,
        "derived_diagnostic_arrays",
        [directions, *case_arrays],
    )
    review = {
        "name": review_archive.name,
        "kind": "compact_review_archive",
        "bytes": review_archive.stat().st_size,
        "sha256": sha256_file(review_archive),
    }
    release_manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "release_tag": f"experiment/{experiment_id}",
        "assets": [derived, review],
    }
    manifest_path = output_directory / f"{run_id}_release-assets.json"
    manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n"
    )
    release_manifest["manifest_asset"] = {
        "name": manifest_path.name,
        "bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }
    return release_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--sweep-id")
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--diagnostic-run-id")
    parser.add_argument("--diagnostic-artifact-directory", type=Path)
    parser.add_argument("--review-archive", type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--manifest-copy", type=Path)
    args = parser.parse_args()
    if args.diagnostic_run_id and args.diagnostic_artifact_directory and args.review_archive and not (
        args.run_id or args.artifact_directory or args.sweep_id or args.artifact_root
    ):
        result = build_diagnostic_release_assets(
            args.experiment_id,
            args.diagnostic_run_id,
            args.diagnostic_artifact_directory,
            args.review_archive,
            args.output_directory,
        )
    elif args.sweep_id and args.artifact_root and not (
        args.run_id or args.artifact_directory or args.diagnostic_run_id
    ):
        result = build_sweep_release_assets(
            args.experiment_id,
            args.sweep_id,
            args.artifact_root,
            args.output_directory,
        )
    elif args.run_id and args.artifact_directory and not (
        args.sweep_id or args.artifact_root or args.diagnostic_run_id
    ):
        result = build_release_assets(
            args.experiment_id,
            args.run_id,
            args.artifact_directory,
            args.output_directory,
        )
    else:
        parser.error(
            "provide either --run-id with --artifact-directory or "
            "--sweep-id with --artifact-root, or the diagnostic arguments"
        )
    if args.manifest_copy:
        source = args.output_directory / result["manifest_asset"]["name"]
        args.manifest_copy.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_copy.write_bytes(source.read_bytes())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
