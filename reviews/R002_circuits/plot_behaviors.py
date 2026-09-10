#!/usr/bin/env python3
"""Scientific figures from verified hard execution and supplied soft trajectories."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import analyze_subset as a
import refine_circuits as r

def main():
    probe=a.load(a.ROOT/"artifacts/reconstructed_fixed_evaluation_set.npz")
    inputs=probe["inputs"].astype(np.uint8);target=probe["target"][0,...,0]
    near=a.NAMES[1];cycle=a.NAMES[-1];gap=a.NAMES[-2]
    nd=r.build(*r.load_rule(near));cd=r.build(*r.load_rule(cycle))
    ns=a.hard_rollout(inputs,nd,26);cs=a.hard_rollout(inputs,cd,24)
    ne=(ns[...,0]!=target).sum(axis=(2,3))
    last=[max(np.flatnonzero(ne[:,i]>0),default=-1) for i in range(32)]
    chosen=max(range(32),key=lambda i:(last[i],ne[20,i]))
    # Independent unsimplified execution of the selected diagnostic probe.
    gates,wires=r.load_rule(near)
    direct=a.base.rollout(inputs[chosen:chosen+1],lambda g:a.base.full_step(g,gates,wires),26)
    assert np.array_equal(direct,ns[:,chosen:chosen+1])
    old=a.load(a.ROOT/"artifacts"/gap/"probe_trajectory_update_400.npz")
    new=a.load(a.ROOT/"artifacts"/gap/"probe_trajectory_update_500.npz")
    cmap=LinearSegmentedColormap.from_list("binary_state",["#ffffff","#183e62"])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":"#17233a","svg.fonttype":"none"})
    fig,axes=plt.subplots(3,4,figsize=(11,9.8))
    fig.subplots_adjust(left=.045,right=.975,top=.855,bottom=.085,hspace=.72,wspace=.12)
    def panel(ax,values,title,caption):
        ax.imshow(values,vmin=0,vmax=1,cmap=cmap,interpolation="nearest")
        ax.set_xticks([]);ax.set_yticks([])
        for s in ax.spines.values():s.set_color("#94a3b8");s.set_linewidth(.6)
        ax.set_title(title,fontsize=10,pad=6)
        ax.set_xlabel(caption,fontsize=8,color="#475569",labelpad=5)
    for j,t in enumerate((20,21,22,25)):
        panel(axes[0,j],ns[t,chosen,...,0],f"Runtime tick {t}",
              f"{int(ne[t].sum())} errors across 32 probes\n{int(ne[t,chosen])} in the displayed grid")
    for j,t in enumerate((20,21,22,23)):
        errors=int(np.sum(cs[t,...,0]!=target))
        panel(axes[1,j],cs[t,0,...,0],f"Runtime tick {t}",f"{errors:,} errors across 32 probes")
    sources=((old,"soft",400),(old,"common",400),(new,"soft",500),(new,"common",500))
    for j,(saved,hard,update) in enumerate(sources):
        values=saved[hard][-1,0,...,0]
        errors=int(np.sum((values>=.5)!=target))
        label="Relaxed output" if hard=="soft" else "Boolean circuit"
        caption=f"{errors} {'thresholded-output' if hard=='soft' else 'Boolean'} errors in this grid"
        panel(axes[2,j],values,f"Update {update}: {label}",caption)
    fig.suptitle("R002 — runtime and rounding reveal different behaviors",x=.045,y=.985,ha="left",fontsize=17,fontweight="bold")
    fig.text(.045,.94,"Frozen update-500 circuits in A/B; optimizer updates 400 and 500 compared in C. No additional training.",fontsize=9,color="#475569")
    fig.text(.045,.902,f"A   Categorical / decay seed 13: a late generator  ·  probe {chosen}, selected for late settling",fontsize=10.5,fontweight="bold")
    fig.text(.045,.613,"B   Direct truth / decay seed 14: exact even-tick output, alternating with another pattern",fontsize=10.5,fontweight="bold")
    fig.text(.045,.327,"C   Direct truth / decay seed 8: soft and hard execution at runtime tick 20  ·  probe 0",fontsize=10.5,fontweight="bold")
    fig.text(.045,.028,"White = 0; navy = 1; intermediate shades are relaxed outputs. Each grid is 16 × 16 with a zero exterior.\nThese selected diagnostics do not change the frozen 20-tick R002 success classifications.",fontsize=8,color="#475569",linespacing=1.5)
    for ext in ("png","svg"):fig.savefig(a.OUT/f"circuit_behaviors.{ext}",dpi=180,facecolor="white")
    plt.close(fig)
    metadata={"near_miss_displayed_probe":chosen,"near_miss_probe_selection":"latest incorrect runtime tick, then maximum errors at tick 20",
              "near_miss_ticks":[20,21,22,25],"cycle_probe":0,"cycle_ticks":[20,21,22,23],"gap_probe":0,"gap_updates":[400,500],"gap_runtime_ticks":20}
    (a.OUT/"figure_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata))

if __name__=="__main__":main()
