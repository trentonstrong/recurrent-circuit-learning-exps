#!/usr/bin/env python3
"""Replay the released seed-0 R004 circuits with pinned NumPy Boolean tools.

Direct LUT execution checks the first probe for 20 ticks; the pinned FactoredDAG
executes all 66 declared starts for 256 ticks. This does not rerun GPU training.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def load(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.bundle.resolve();ref=args.reference.resolve()
    sys.path.insert(0,str(ref))
    from scripts import r002_runtime_profile as p
    args.output.mkdir(parents=True,exist_ok=True)
    probe=load(root/'fixed_evaluation_set.npz')
    assert hashlib.sha256((root/'fixed_evaluation_set.npz').read_bytes()).hexdigest()=='d49ecf23fd932fba25b1068f8c90820265a05fd90daadc372089520889090b2f'
    initial=probe['inputs'].astype(np.uint8);target=probe['target'][0,...,0].astype(np.uint8)
    fresh=np.random.Generator(np.random.PCG64(20260910)).integers(0,2,size=(32,16,16,8),dtype=np.uint8)
    assert hashlib.sha256(fresh.tobytes()).hexdigest()=='0bd05a2d3f281ff9f5e26c118677ba91ff9fc9c3bf777eed4ede6bca60387d65'
    starts=np.concatenate([initial,fresh,np.zeros((1,16,16,8),np.uint8),np.ones((1,16,16,8),np.uint8)])
    with gzip.open(root/'results/R004_formal_attempt01_analysis_attempt03/runtime_profiles.jsonl.gz','rt') as f:
        records=[json.loads(l) for l in f]
    selected=[r for r in records if r['seed']==0]
    assert len(selected)==32
    cache={};out=[];started=time.perf_counter()
    for record in selected:
        c,k,h=record['condition'],record['global_update'],record['hardening']
        if k==500:
            folder=root/'source'/f'R002_formal_attempt01_seed00_{c}'
            export=folder/f'final_{h}_circuit.npz'
            trajectory=folder/'probe_trajectory_update_500.npz'
        else:
            folder=root/'continued'/f'R004_formal_attempt01_seed00_{c}'
            export=folder/f'circuit_update_{k}_{h}.npz'
            trajectory=folder/f'probe_trajectory_update_{k}.npz'
        assert hashlib.sha256(export.read_bytes()).hexdigest()==record['source']['artifact']['sha256']
        stored=load(export)
        original_gates=[stored[f'gate_{label}'] for label in p.LABELS]
        assert all(np.issubdtype(a.dtype,np.integer) and np.all((a>=0)&(a<=15)) for a in original_gates)
        gates=[a.astype(np.uint8) for a in original_gates]
        wires=[(stored[f'wire_{label}_a'],stored[f'wire_{label}_b']) for label in p.LABELS]
        assert p.subset.tree_digest([a for pair in wires for a in pair])==record['source']['wiring_sha256']
        rule_hash=p.subset.tree_digest(gates+[a for pair in wires for a in pair])
        assert rule_hash==record['source']['rule_sha256']
        reused=rule_hash in cache
        if not reused:
            dag=p.refine.build(gates,wires)
            structure=p.compact_structure(p.subset.dag_summary(dag))
            source_states=p.subset.base.rollout(initial[:1],lambda x:p.subset.base.full_step(x,gates,wires),20)
            state=starts.copy();errors=[];first_states=[]
            full_changes=[];visible_changes=[]
            previous=None
            for tick in range(257):
                errors.append(np.sum(state[...,0]!=target,axis=(1,2)))
                if tick<=20:first_states.append(state[:1].copy())
                if previous is not None:
                    full_changes.append(np.sum(state!=previous,axis=(1,2,3)))
                    visible_changes.append(np.sum(state[...,0]!=previous[...,0],axis=(1,2)))
                if tick<256:
                    previous=state
                    state=dag.evaluate(p.subset.base.patches(state))
            first_states=np.stack(first_states)
            assert np.array_equal(source_states,first_states)
            certificate=p.universal_certificate(dag,target,256)
            cache[rule_hash]={'structure':structure,'errors':np.stack(errors),'first_states':first_states,'certificate':certificate,
                'full_changes':np.stack(full_changes),'visible_changes':np.stack(visible_changes)}
        value=cache[rule_hash]
        assert value['structure']==record['structure'],(c,k,h,'structure')
        assert value['certificate']==record['universal_state_certificate'],(c,k,h,'certificate')
        actual=value['errors'];assert actual.shape==(257,66)
        assert np.array_equal(actual[20],[t['errors_at_20'] for t in record['trajectories']])
        for group,sl in [('original_probe',slice(0,32)),('fresh_probe',slice(32,64)),('all_zero',slice(64,65)),('all_one',slice(65,66))]:
            reported=record['initialization_groups'][group]
            assert np.array_equal(actual[:,sl].sum(1),reported['aggregate_errors_by_tick']),(c,k,h,group)
            assert np.array_equal((actual[:,sl]==0).sum(1),reported['perfect_initializations_by_tick']),(c,k,h,group)
        saved=load(trajectory)
        assert np.array_equal(value['first_states'],saved[h]),(c,k,h,'saved_trajectory')
        out.append({'condition':c,'update':k,'hardening':h,'rule_sha256':rule_hash,'reused':reused,
                    'original_errors_20':int(actual[20,:32].sum()),'fresh_errors_20':int(actual[20,32:64].sum()),
                    'zero_errors_20':int(actual[20,64]),'one_errors_20':int(actual[20,65]),
                    'zero_errors_21':int(actual[21,64]),'zero_errors_22':int(actual[22,64]),
                    'visible_core_binary_nodes':value['structure']['visible_core_binary_nodes'],
                    'universal_status':value['certificate']['status'],
                    'universal_onset':value['certificate']['certified_visible_settling_tick']})
        print(f'{c} k={k} {h}: passed'+(' (reused)' if reused else ''),flush=True)
    result={'status':'passed','logical_modes':len(out),'logical_trajectories':len(out)*66,'unique_rules':len(cache),
       'runtime_ticks':256,'direct_lut_first_probe_ticks':20,'verified_saved_trajectories':len(out),
       'scope':'Only the fixed seed-0 subset is replayed. Other seeds are covered by record-level checks, not direct circuit replay.',
       'wall_seconds':time.perf_counter()-started,'cases':out}
    (args.output/'subset_replay.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
