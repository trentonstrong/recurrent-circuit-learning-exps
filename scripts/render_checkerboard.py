#!/usr/bin/env python3
"""Render an R001 probe trajectory as a compact SVG contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def default_ticks(tick_count: int) -> list[int]:
    """Choose up to five evenly spaced ticks, including both endpoints."""
    if tick_count < 1:
        raise ValueError("trajectory must contain at least one tick")
    return sorted({round(index * (tick_count - 1) / 4) for index in range(5)})


def parse_ticks(value: str | None, tick_count: int) -> list[int]:
    if value is None:
        return default_ticks(tick_count)
    if value == "all":
        return list(range(tick_count))
    try:
        ticks = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("ticks must be 'all' or comma-separated integers") from error
    if not ticks or any(tick < 0 or tick >= tick_count for tick in ticks):
        raise ValueError(f"ticks must be between 0 and {tick_count - 1}")
    return ticks


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = {"soft", "hard"}.difference(archive.files)
        if missing:
            raise ValueError(
                f"trajectory is missing arrays: {', '.join(sorted(missing))}"
            )
        arrays = {name: np.asarray(archive[name]) for name in ("soft", "hard")}
    if any(array.ndim != 5 for array in arrays.values()):
        raise ValueError(
            "trajectory arrays must have shape (tick, sample, y, x, channel)"
        )
    if arrays["soft"].shape != arrays["hard"].shape:
        raise ValueError("soft and hard trajectory shapes differ")
    return arrays


def render_svg(
    arrays: dict[str, np.ndarray],
    *,
    title: str,
    modes: list[str],
    ticks: list[int],
    sample: int,
    channel: int,
) -> str:
    shape = arrays["soft"].shape
    _, sample_count, grid_height, grid_width, channel_count = shape
    if not 0 <= sample < sample_count:
        raise ValueError(f"sample must be between 0 and {sample_count - 1}")
    if not 0 <= channel < channel_count:
        raise ValueError(f"channel must be between 0 and {channel_count - 1}")

    cell = 8
    panel_width = grid_width * cell
    panel_height = grid_height * cell
    left = 58
    top = 58
    gap_x = 14
    gap_y = 30
    width = left + len(ticks) * (panel_width + gap_x) + 10
    height = top + len(modes) * (panel_height + gap_y) + 10

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        + f'viewBox="0 0 {width} {height}">',
        f"  <title>{escape(title)}</title>",
        '  <rect width="100%" height="100%" fill="#f7f7f5"/>',
        '  <g font-family="ui-monospace, monospace" fill="#222">',
        f'    <text x="10" y="18" font-size="13" font-weight="bold">{escape(title)}</text>',
        f'    <text x="10" y="35" font-size="10">sample {sample}, channel {channel}</text>',
    ]
    for column, tick in enumerate(ticks):
        x = left + column * (panel_width + gap_x)
        lines.append(
            f'    <text x="{x + panel_width / 2:g}" y="50" font-size="10" '
            f'text-anchor="middle">tick {tick}</text>'
        )
    for row, mode in enumerate(modes):
        y = top + row * (panel_height + gap_y)
        lines.append(
            f'    <text x="48" y="{y + panel_height / 2:g}" font-size="11" '
            f'text-anchor="end" dominant-baseline="middle">{mode}</text>'
        )
        for column, tick in enumerate(ticks):
            x = left + column * (panel_width + gap_x)
            values = np.clip(arrays[mode][tick, sample, :, :, channel], 0.0, 1.0)
            lines.append(
                f'    <g transform="translate({x} {y})" shape-rendering="crispEdges">'
            )
            lines.append(
                f'      <rect width="{panel_width}" height="{panel_height}" fill="white" '
                'stroke="#777"/>'
            )
            for grid_y, grid_x in np.ndindex(values.shape):
                gray = round(255 * (1.0 - float(values[grid_y, grid_x])))
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                lines.append(
                    f'      <rect x="{grid_x * cell}" y="{grid_y * cell}" '
                    f'width="{cell}" height="{cell}" fill="{color}"/>'
                )
            lines.append("    </g>")
    lines.extend(["  </g>", "</svg>", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("soft", "hard", "both"), default="both")
    parser.add_argument("--ticks", help="comma-separated tick indices, or 'all'")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args()

    arrays = load_trajectory(args.trajectory)
    ticks = parse_ticks(args.ticks, arrays["soft"].shape[0])
    modes = ["soft", "hard"] if args.mode == "both" else [args.mode]
    output = args.output or args.trajectory.with_suffix(".svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_svg(
            arrays,
            title=args.trajectory.stem,
            modes=modes,
            ticks=ticks,
            sample=args.sample,
            channel=args.channel,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
