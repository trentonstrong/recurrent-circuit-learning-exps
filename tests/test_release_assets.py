from __future__ import annotations

import json
import tarfile
from pathlib import Path

from scripts.build_release_assets import (
    build_release_assets,
    build_sweep_release_assets,
    sha256_file,
)


def test_release_assets_embed_experiment_and_run_identity(tmp_path: Path) -> None:
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    for name, value in (
        ("checkpoint_update_000.npz", b"checkpoint"),
        ("probe_trajectory_update_000.npz", b"trajectory"),
        ("fixed_evaluation_set.npz", b"evaluation"),
        ("final_hard_circuit.npz", b"circuit"),
    ):
        (artifact_directory / name).write_bytes(value)

    output_directory = tmp_path / "release"
    first = build_release_assets(
        "R001", "R001_test", artifact_directory, output_directory
    )
    checkpoint_asset = output_directory / "R001_test_checkpoints.tar.gz"
    first_digest = sha256_file(checkpoint_asset)
    second = build_release_assets(
        "R001", "R001_test", artifact_directory, output_directory
    )

    assert first["release_tag"] == "experiment/R001"
    assert first_digest == sha256_file(checkpoint_asset)
    assert first["assets"] == second["assets"]
    with tarfile.open(checkpoint_asset, "r:gz") as archive:
        members = archive.getmembers()
        assert all(member.mtime == 0 for member in members)
        contents = json.load(archive.extractfile("CONTENTS.json"))  # type: ignore[arg-type]
    assert contents["experiment_id"] == "R001"
    assert contents["run_id"] == "R001_test"
    assert contents["bundle_kind"] == "checkpoints"


def test_sweep_release_assets_preserve_run_directories(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    for condition in ("categorical", "truth"):
        run_directory = artifact_root / f"R002_test_seed00_{condition}"
        run_directory.mkdir(parents=True)
        for name, value in (
            ("checkpoint_update_000.npz", b"checkpoint"),
            ("probe_trajectory_update_000.npz", b"trajectory"),
            ("final_native_circuit.npz", b"native"),
            ("final_common_circuit.npz", b"common"),
        ):
            (run_directory / name).write_bytes(value)

    output_directory = tmp_path / "release"
    first = build_sweep_release_assets(
        "R002", "test", artifact_root, output_directory
    )
    checkpoint_asset = output_directory / "R002_test_checkpoints.tar.gz"
    first_digest = sha256_file(checkpoint_asset)
    second = build_sweep_release_assets(
        "R002", "test", artifact_root, output_directory
    )

    assert first["release_tag"] == "experiment/R002"
    assert first_digest == sha256_file(checkpoint_asset)
    assert first["assets"] == second["assets"]
    with tarfile.open(checkpoint_asset, "r:gz") as archive:
        members = archive.getmembers()
        assert all(member.mtime == 0 for member in members)
        contents = json.load(archive.extractfile("CONTENTS.json"))  # type: ignore[arg-type]
        paths = {record["path"] for record in contents["files"]}
    assert paths == {
        "R002_test_seed00_categorical/checkpoint_update_000.npz",
        "R002_test_seed00_truth/checkpoint_update_000.npz",
    }
    assert contents["experiment_id"] == "R002"
    assert contents["sweep_id"] == "test"
    assert contents["bundle_kind"] == "checkpoints"
