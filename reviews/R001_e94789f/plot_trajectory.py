"""Plot the committed R001 records; performs no model execution or training.

Requires matplotlib. Run from any directory with this repository checked out.
"""

from pathlib import Path
import hashlib
import json
import math
import platform
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "results/R001_20260909T231016Z_seed23_jaxgpu_attempt01"
manifest_path = RUN / "manifest.json"
metrics_path = RUN / "metrics.jsonl"
manifest = json.loads(manifest_path.read_text())
rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
assert len(rows) == 500
assert all(row["update"] == index + 1 for index, row in enumerate(rows))
assert all(math.isfinite(value) for row in rows for value in row.values())
evaluations = manifest["evaluations"]
assert [item["update"] for item in evaluations] == list(range(0, 501, 50))

# A record labeled update u measures the parameters before that update.
k_train = [row["update"] - 1 for row in rows]
k_probe = [item["update"] for item in evaluations]
train_bits = 2 * 16 * 16
probe_bits = 32 * 16 * 16
first_exact = next(item["update"] for item in evaluations if item["exact_all_32"])
assert all(item["exact_all_32"] for item in evaluations if item["update"] >= first_exact)
max_gradient = max(row["gradient_max_abs"] for row in rows)
assert max_gradient < 100.0

analysis = {
    "reviewed_commit": "e94789f64a5a9212c3540ce49ee928ca952998e0",
    "scope": "Analysis of committed records and source; no training or checkpoint replay",
    "inputs": {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (manifest_path, metrics_path)
    },
    "record_count": len(rows),
    "first_exact_saved_probe_update": first_exact,
    "final_probe_hard_bit_errors": evaluations[-1]["hard_bit_errors"],
    "final_probe_soft_sse": evaluations[-1]["soft_terminal_summed_squared_error"],
    "final_probe_soft_mse": evaluations[-1]["soft_terminal_summed_squared_error"] / probe_bits,
    "first_zero_training_batch_error_record": next(row["update"] for row in rows if row["pre_update_hard_loss"] == 0),
    "maximum_raw_gradient_entry": max_gradient,
    "updates_exceeding_clip_bound": sum(row["gradient_max_abs"] > 100 for row in rows),
    "summed_update_seconds": sum(row["synchronized_seconds"] for row in rows),
    "median_update_seconds": statistics.median(row["synchronized_seconds"] for row in rows),
    "peak_allocator_gib_reported": manifest["device_memory"]["peak_bytes_in_use"] / 2**30,
    "plotting_environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__},
}
assert math.isclose(analysis["summed_update_seconds"], manifest["synchronized_update_seconds"], abs_tol=1e-9)
assert math.isclose(analysis["median_update_seconds"], manifest["median_update_seconds"], abs_tol=1e-9)
(HERE / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#252525", "text.color": "#252525",
    "svg.fonttype": "none", "svg.hashsalt": "R001-e94789f",
})
fig, axes = plt.subplots(3, 1, figsize=(9.4, 8.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.3, 1, 1.2]})
fig.subplots_adjust(left=0.11, right=0.98, top=0.87, bottom=0.17, hspace=0.26)
fig.suptitle("R001: a long plateau precedes exact hard readout", x=0.11, y=0.969,
             ha="left", fontsize=16, fontweight="bold")
fig.text(0.11, 0.932, "Seed 23  |  500 optimizer updates  |  16 x 16 grids  |  Channel 0 at tick 20",
         fontsize=10.5, color="#555555")
blue, orange, gray = "#28649A", "#C64E17", "#777777"
for ax in axes:
    ax.grid(axis="y", alpha=0.15)
    ax.axvline(first_exact, color="#648D70", linestyle=":", linewidth=1.2)
    ax.set_xlim(0, 505)

axes[0].semilogy(k_train, [r["pre_update_soft_loss"] / train_bits for r in rows],
                 color=blue, linewidth=1.1, label="Fresh training batches (2 grids)")
axes[0].semilogy(k_probe, [e["soft_terminal_summed_squared_error"] / probe_bits for e in evaluations],
                 color=orange, linewidth=1.3, marker="o", markersize=3.5,
                 label="Fixed probe (32 grids)")
axes[0].axhline(0.25, color=gray, linestyle="--", linewidth=0.8,
                label="Constant 0.5 prediction")
axes[0].set_ylabel("Soft mean squared error")
axes[0].legend(loc="upper right", fontsize=8.5, frameon=True,
               facecolor="white", edgecolor="none", framealpha=0.95)

axes[1].plot(k_train, [r["pre_update_hard_loss"] / train_bits for r in rows],
             color=blue, linewidth=1, label="Fresh training batches")
axes[1].plot(k_probe, [e["hard_bit_errors"] / probe_bits for e in evaluations],
             color=orange, linewidth=1.4, marker="o", markersize=3.5,
             label="Fixed probe")
axes[1].set_ylabel("Hard bit error rate")
axes[1].yaxis.set_major_formatter(PercentFormatter(1))
axes[1].set_ylim(-0.015, 0.55)
axes[1].set_yticks([0, 0.25, 0.5])
axes[1].text(272, 0.21, "First exact saved probe: update 250\n32/32 grids remain exact through 500",
             fontsize=9, color="#356445")
axes[1].legend(loc="upper right", fontsize=8.5, frameon=False)

axes[2].semilogy(k_train, [r["gradient_l2"] for r in rows], color=blue,
                 linewidth=1.1, label="Gradient L2 norm")
axes[2].semilogy(k_train, [r["gradient_max_abs"] for r in rows], color=orange,
                 linewidth=1.1, label="Largest absolute gradient entry")
axes[2].axhline(100, color=gray, linestyle="--", linewidth=0.8,
                label="Elementwise clip bound (100)")
axes[2].set_ylabel("Raw gradient magnitude")
axes[2].set_xlabel("Optimizer updates already applied, k")
axes[2].legend(loc="upper left", fontsize=8.2, frameon=True, ncol=1,
               facecolor="white", edgecolor="none", framealpha=0.95)
axes[2].set_ylim(0.002, 250)
axes[2].set_xticks(list(range(0, 501, 50)))

fig.text(0.11, 0.042,
         "Training record u is plotted at k = u - 1; probe checkpoints are plotted after their stated update.\n"
         "All 500 training records and 11 saved evaluations are shown. Probe connectors guide the eye;\n"
         "intermediate probe accuracy was not measured. Curves are one seed, with no smoothing or uncertainty bands.",
         fontsize=8.5, color="#555555", linespacing=1.4)
fig.savefig(HERE / "trajectory.svg", metadata={"Date": None})
fig.savefig(HERE / "trajectory.png", dpi=170, metadata={"Software": "Matplotlib"})
plt.close(fig)
print(json.dumps(analysis, indent=2))
