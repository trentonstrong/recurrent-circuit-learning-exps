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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-directory", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    result = build_release_assets(
        args.experiment_id,
        args.run_id,
        args.artifact_directory,
        args.output_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
