#!/usr/bin/env python3
"""Plot logged R002 trajectories; run analyze_results.py first."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/"plot_data.json").read_text())
styles = (
    ("categorical_reference_decay", "Categorical · decay 0.01", "#2563a6", "-"),
    ("categorical_no_decay", "Categorical · no decay", "#2563a6", "--"),
    ("truth_reference_decay", "Direct truth · decay 0.01", "#c66024", "-"),
    ("truth_no_decay", "Direct truth · no decay", "#c66024", "--"),
)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#334155", "text.color": "#17233a",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "axes.edgecolor": "#cbd5e1", "svg.fonttype": "none",
})
fig, axes = plt.subplots(2,2,figsize=(12,8))
fig.subplots_adjust(left=.085, right=.97, bottom=.105, top=.80, hspace=.45, wspace=.28)
panels = [
    ("soft_sse", "A   Relaxed output loss", "Soft SSE on 8,192 visible bits", 4000),
    ("common_errors", "B   Boolean output errors", "Common-hard errors out of 8,192", 4400),
    ("q_path_length", "C   Travel through effective truth tables", "Cumulative sum of per-update L2 changes", None),
    ("common_gate_change_events", "D   Changes in rounded Boolean gates", "Cumulative gate-ID change events", None),
]
for ax,(key,title,ylabel,upper) in zip(axes.flat,panels):
    for condition,label,color,line in styles:
        arr = np.asarray(data[condition][key])
        x = np.asarray(data[condition]["saved_updates"]) if arr.shape[1]==11 else np.arange(1,501)
        ax.plot(x,np.median(arr,axis=0),color=color,ls=line,lw=2.35,label=label)
    ax.set_title(title,loc="left",pad=12,fontweight="bold",fontsize=11)
    ax.set_xlabel("Optimizer update")
    ax.set_ylabel(ylabel,fontsize=9)
    ax.set_xlim(0,500)
    ax.set_ylim(bottom=0,top=upper)
    ax.set_xticks(np.arange(0,501,100))
    ax.grid(axis="y",alpha=.24,lw=.7)
axes[0,0].axhline(2048,color="#64748b",ls=":",lw=1)
axes[0,0].text(495,2120,"Constant 0.5 output",ha="right",va="bottom",fontsize=8,color="#64748b")
fig.suptitle("R002 — paired training trajectories",x=.085,y=.965,ha="left",fontsize=19,fontweight="bold")
fig.text(.085,.917,"Medians over 16 paired seeds per condition · common multilinear kernel · fixed 500-update budget",fontsize=10,color="#475569")
handles,labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="upper left",bbox_to_anchor=(.076,.895),ncol=2,frameon=False,fontsize=9,columnspacing=2.8)
fig.text(.085,.035,"Gate changes count repeated slot-level flips, not distinct circuits or neutral paths. Curves summarize seeds; they are not individual trajectories.",fontsize=8.1,color="#475569")
for ext in ("png","svg"):
    fig.savefig(HERE/f"trajectories.{ext}",dpi=180,facecolor="white")
plt.close(fig)
print(HERE/"trajectories.png")
