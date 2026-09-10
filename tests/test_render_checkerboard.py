from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.render_checkerboard import default_ticks, load_trajectory, render_svg


def test_render_checkerboard_trajectory(tmp_path: Path) -> None:
    soft = np.zeros((21, 1, 2, 2, 1), dtype=np.float32)
    soft[20, 0, :, :, 0] = [[0.0, 1.0], [1.0, 0.5]]
    path = tmp_path / "probe.npz"
    np.savez(path, soft=soft, hard=(soft >= 0.5).astype(np.float32))

    arrays = load_trajectory(path)
    ticks = default_ticks(arrays["soft"].shape[0])
    svg = render_svg(
        arrays,
        title="probe",
        modes=["soft", "hard"],
        ticks=ticks,
        sample=0,
        channel=0,
    )

    assert ticks == [0, 5, 10, 15, 20]
    assert "tick 20" in svg
    assert ">soft</text>" in svg
    assert ">hard</text>" in svg
    assert 'fill="#000000"' in svg
    assert 'fill="#ffffff"' in svg


def test_render_rejects_out_of_range_sample(tmp_path: Path) -> None:
    arrays = {
        "soft": np.zeros((1, 1, 2, 2, 1)),
        "hard": np.zeros((1, 1, 2, 2, 1)),
    }
    with pytest.raises(ValueError, match="sample must be between"):
        render_svg(
            arrays,
            title="probe",
            modes=["hard"],
            ticks=[0],
            sample=1,
            channel=0,
        )
