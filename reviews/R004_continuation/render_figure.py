#!/usr/bin/env python3
"""Render the audited saved-checkpoint trajectories with Matplotlib."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.review/'review_metrics.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.6))
    styles=[('categorical_no_decay','Categorical, no decay','#157e89','-'),
            ('categorical_reference_decay','Categorical, decay','#157e89','--'),
            ('truth_no_decay','Direct truth, no decay','#bb641b','-'),
            ('truth_reference_decay','Direct truth, decay','#bb641b','--')]
    for c,label,color,ls in styles:
        r=report['condition_summary'][c]
        axes[0].plot(r['updates'],r['common_success_trajectory'],label=label,color=color,linestyle=ls,linewidth=1.7,marker='o',markersize=2.7)
        ks=sorted(int(k) for k in r['common_sizes'])
        axes[1].plot(ks,[r['common_sizes'][str(k)]['median'] for k in ks],label=label,color=color,linestyle=ls,linewidth=1.7,marker='o',markersize=4)
    axes[0].set(title='Exact Boolean readouts',ylabel='Successful runs / 16',ylim=(-0.2,16.5),yticks=[0,4,8,12,16],xlim=(0,2025))
    axes[0].axvline(500,color='#808080',linewidth=.9,linestyle=':')
    axes[0].text(525,15.4,'Original budget',fontsize=9,color='#666666')
    axes[1].set(title='Extracted circuit size',ylabel='Median visible-core binary operations',yscale='log',xlim=(475,2025))
    axes[1].set_yticks([10,30,100,300]);axes[1].set_yticklabels(['10','30','100','300'])
    for ax in axes:
        ax.set_xlabel('Optimizer update')
        ax.grid(axis='y',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True)
    fig.suptitle('R004: more solutions, larger extracted circuits',fontsize=14,y=.98)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=2,loc='lower center',bbox_to_anchor=(.5,.025),frameon=False,fontsize=9.5)
    fig.tight_layout(rect=(0,.15,1,.94))
    fig.savefig(args.review/'trajectories.svg',metadata={'Date':None})
    fig.savefig(args.review/'trajectories_preview.png',dpi=155)


if __name__=='__main__':main()
