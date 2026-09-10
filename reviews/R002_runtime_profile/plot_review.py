#!/usr/bin/env python3
"""Research figure from the independently audited R002 records."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OUT = Path(__file__).resolve().parent
data = json.loads((OUT / "audit.json").read_text())
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                     "axes.spines.top":False,"axes.spines.right":False})
fig = plt.figure(figsize=(11,8.5),facecolor="white")
grid = fig.add_gridspec(2,1,height_ratios=[1.6,1],left=.22,right=.96,top=.83,bottom=.19,hspace=.68)
ax = fig.add_subplot(grid[0])
conditions = ["categorical_reference_decay","categorical_no_decay","truth_reference_decay","truth_no_decay"]
colors = ["#245475","#245475","#c26b24","#c26b24"]
for i,(condition,color) in enumerate(zip(conditions,colors)):
    modes = sorted([m for m in data["primary_modes"] if m["condition"]==condition],key=lambda m:m["seed"])
    sizes = np.array([m["visible_core_binary_nodes"] for m in modes])
    jitter = ((np.arange(16)*7)%16-7.5)*.022
    ax.scatter(i+jitter,sizes,s=30,color=color,alpha=.9,zorder=3)
    median = float(np.median(sizes))
    ax.plot([i-.27,i+.27],[median,median],color="#17273d",lw=2.3,zorder=4)
    ax.text(i+.29,median,f"{median:g}",va="center",fontsize=10,fontweight="bold",color="#17273d")
ax.set_xticks(range(4),["Categorical\nDecay 0.01","Categorical\nNo decay","Direct truth\nDecay 0.01","Direct truth\nNo decay"])
ax.set_xlim(-.55,3.65);ax.set_ylim(0,250)
ax.set_yticks([0,50,100,150,200,250]);ax.grid(axis="y",alpha=.18)
ax.set_ylabel("Binary operations in visible core")
ax.set_title("A   Every direct-truth circuit is larger under this simplifier",loc="left",pad=17,fontweight="bold")
ax.text(.02,.93,"Dots: all 16 seeds per condition\nHorizontal marks: medians",transform=ax.transAxes,va="top",fontsize=10,color="#475569")

ax = fig.add_subplot(grid[1])
examples = {item["seed"]:item for item in data["phase_examples"]}
rows=[]
for seed in (2,14):
    values=examples[seed]["perfect_counts_at_ticks_64_through_79"]
    rows.extend([np.array(values["original_probe"])/32,np.array(values["all_zero"])])
ax.imshow(rows,cmap=ListedColormap(["#e9eef3","#245475"]),vmin=0,vmax=1,aspect="auto",interpolation="nearest")
ax.set_xticks(range(16),range(64,80));ax.set_xlabel("Runtime tick of the frozen circuit")
ax.set_yticks(range(4),["Seed 2: 32 original probes","Seed 2: zero initialization","Seed 14: 32 original probes","Seed 14: zero initialization"])
ax.set_xticks(np.arange(-.5,16,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
ax.grid(which="minor",color="white",linewidth=1.5);ax.tick_params(which="minor",bottom=False,left=False)
ax.set_title("B   A correct readout can select one phase of a cycle",loc="left",pad=16,fontweight="bold")
for spine in ax.spines.values():spine.set_visible(False)
fig.text(.035,.955,"R002 — circuit size and observation phase",fontsize=21,fontweight="bold",color="#17273d")
fig.text(.035,.916,"Common hardening at optimizer update 500. No additional training.",fontsize=12,color="#475569")
fig.text(.035,.092,"Blue: all starts in that row are correct; gray: none are correct. Both examples use direct truth with decay 0.01.",fontsize=10,color="#475569")
fig.text(.035,.069,"Seed 2 has full-state period 8 and visible period 4; seed 14 has period 2. Phases follow the recorded exact cycles.",fontsize=10,color="#475569")
fig.text(.035,.035,"Operation counts use the same exact rewriting method for all circuits. They exclude wiring and state cost and are not proven minima.",fontsize=9,color="#475569")
fig.savefig(OUT/"size_and_phase.png",dpi=180)
fig.savefig(OUT/"size_and_phase.svg")
plt.close(fig)
