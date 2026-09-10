#!/usr/bin/env python3
"""Render figures from the post-hoc JSON and verified stored probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = report["checkpoints"]
    updates = [r["update"] for r in rows]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12,
                         "axes.labelcolor": "#273340", "text.color": "#172738",
                         "axes.edgecolor": "#aab5bf", "svg.fonttype": "none"})
    blue, orange, teal, gray = "#2466a8", "#c9602f", "#218a83", "#6f7c87"
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    fig.subplots_adjust(top=.85, bottom=.13, hspace=.5, wspace=.28)
    fig.suptitle("R001: a small rule emerges inside a large representation", fontsize=18, x=.07, ha="left", y=.975)
    fig.text(.07, .93, "Seed 23  •  3,040 gate slots  •  32 fixed evaluation grids  •  20 synchronous ticks", fontsize=11, color=gray)
    ax = axes[0, 0]
    ax.plot(updates, [r["native"]["hard_bit_errors"] / 8192 for r in rows], "o-", color=blue, label="Hard bit-error fraction")
    ax.plot(updates, [r["recorded_soft_terminal_summed_squared_error"] / 8192 for r in rows], "s--", color=teal, label="Soft mean squared error", markersize=4)
    ax.set(title="A  Terminal fit", ylabel="Error per visible bit", ylim=(-.02, .53))
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.annotate("32/32 exact", (250, 0), xytext=(280, .16), arrowprops={"arrowstyle": "->", "color": blue}, color=blue)
    ax = axes[0, 1]
    ax.plot(updates, [r["non_A_slots"] for r in rows], "o-", color=gray, label="Gate slots changed from A")
    ax.plot(updates, [r["simplified"]["and_nodes"] + r["simplified"]["xor_nodes"] for r in rows], "s-", color=orange, label="Simplified binary operations")
    ax.set(title="B  Representation versus simplified rule", ylabel="Count", ylim=(-1, 37))
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(498, 33.8, "33", ha="right", color=gray)
    ax.text(498, 5.1, "4", ha="right", color=orange)
    ax = axes[1, 0]
    ax.plot(updates, [r["mean_gate_entropy_bits"] for r in rows], "o-", color=teal)
    ax.set(title="C  Mixtures become less concentrated", ylabel="Mean categorical entropy (bits/gate)", ylim=(0, .12))
    ax.text(18, .104, "Entropy rises while fit improves.\nThis is not a description-length score.", fontsize=9, color=gray)
    ax = axes[1, 1]
    intervals = report["intervals"]
    for item in intervals:
        same = item["exact_local_rule_disagreement"]["any_channel_disagreement_fraction"] == 0
        center = (item["from_update"] + item["to_update"]) / 2
        ax.bar(center, item["native_gate_flips"], width=36, color=blue if same else orange)
    ax.set(title="D  Gate changes between saved checkpoints", ylabel="Changed native gate IDs", ylim=(0, 15))
    ax.legend(handles=[Patch(color=blue, label="Same full Boolean update rule"),
                       Patch(color=orange, label="Different Boolean update rule")], frameon=False, fontsize=9, loc="upper right")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update")
        ax.set_xlim(-8, 510)
        ax.set_xticks(range(0, 501, 100))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.text(.07, .045, "Simplification propagates constants/copies and merges AND/XOR expressions with complemented edges.\nCounts omit routing, storage, boundaries and execution protocol. Checkpoints do not reveal every intervening step.", fontsize=9, color=gray)
    save(fig, args.output_dir, "checkpoint_summary")

    selected = [0, 150, 200, 250, 500]
    columns = [("soft", 20)] + [("hard", t) for t in [4, 8, 12, 16, 20]]
    fig, axes = plt.subplots(len(selected), len(columns), figsize=(11.5, 9.5))
    fig.subplots_adjust(top=.86, bottom=.08, left=.13, right=.97, wspace=.08, hspace=.17)
    fig.suptitle("The same initial grid, viewed across learning and runtime", fontsize=17, x=.06, ha="left", y=.97)
    fig.text(.06, .925, "Visible channel 0  •  first fixed probe  •  black = 0, white = 1  •  orange outline marks the 16 × 16 grid", fontsize=10, color=gray)
    for i, update in enumerate(selected):
        with np.load(args.artifact_dir / "analysis" / f"probe_trajectory_update_{update:03d}.npz", allow_pickle=False) as probe:
            for j, (mode, tick) in enumerate(columns):
                ax = axes[i, j]
                values = np.pad(probe[mode][tick, 0, ..., 0], 1, constant_values=0)
                ax.imshow(values, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.add_patch(Rectangle((.5, .5), 16, 16, fill=False, edgecolor=orange, linewidth=.8))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if i == 0:
                    ax.set_title(f"{mode.capitalize()} at tick {tick}", fontsize=10, pad=10)
                if j == 0:
                    ax.set_ylabel(f"Update {update}", fontsize=11, rotation=0, labelpad=39, va="center")
    fig.text(.13, .035, "Exterior bits are fixed at zero. The trained checkpoints shown at 250 and 500 reach the checkerboard by tick 16\non all 32 fixed probes; only one probe is pictured. Different intermediate paths can share the same terminal output.", fontsize=9, color=gray)
    save(fig, args.output_dir, "probe_states")


if __name__ == "__main__":
    main()
