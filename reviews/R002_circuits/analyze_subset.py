#!/usr/bin/env python3
"""Independent NumPy hard-circuit review of the uploaded six-run R002 subset."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from reconstruct_probe import reconstruct

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("r001_dag",ROOT/"reviews/R001_checkpoints/analyze_checkpoints.py")
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
NAMES=[
"R002_formal_attempt01_seed04_categorical_reference_decay",
"R002_formal_attempt01_seed13_categorical_reference_decay",
"R002_formal_attempt01_seed04_categorical_no_decay",
"R002_formal_attempt01_seed15_categorical_no_decay",
"R002_formal_attempt01_seed08_truth_reference_decay",
"R002_formal_attempt01_seed14_truth_reference_decay"]
LABELS=base.LAYERS
T=base.T
ENUM_CAP=16

def load(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}

def tree_digest(arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.asarray(a);h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def decode(checkpoint,rep):
    zs=[checkpoint[f"param_{i:03d}"] for i in range(19)]
    qs=[];rounded=[];native=[];round32=[]
    for z in zs:
        z64=z.astype(np.float64)
        if rep=="categorical":
            p=np.exp(z64-z64.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            q=p@T
            p32=np.exp(z-z.max(-1,keepdims=True));p32/=p32.sum(-1,keepdims=True)
            q32=p32@T.astype(np.float32)
            n=z.argmax(-1).astype(np.uint8)
        else:
            q=1/(1+np.exp(-z64));q32=1/(1+np.exp(-z))
            n=((q32>=.5)*[8,4,2,1]).sum(-1).astype(np.uint8)
        qs.append(q)
        rounded.append(((q>=.5)*[8,4,2,1]).sum(-1).astype(np.uint8))
        round32.append(((q32>=.5)*[8,4,2,1]).sum(-1).astype(np.uint8))
        native.append(n)
    assert all(np.array_equal(a,b) for a,b in zip(rounded,round32)),"rounding depends on NumPy diagnostic precision"
    return qs,{"native":native,"common":rounded}

def project(d,outputs):
    p=copy.copy(d);p.outputs=list(outputs);return p

def core_channels(d):
    c={0}
    while True:
        reach=d.reachable([d.outputs[i] for i in c])
        new=c|{d.nodes[i][1]%8 for i in reach if d.nodes[i][0]=="input"}
        if new==c:return sorted(c)
        c=new

def expressions(d,cap=2000):
    memo={}
    def expr(lit):
        if lit in memo:return memo[lit]
        i,inv=divmod(int(lit),2);op,*a=d.nodes[i]
        if op=="constant":return str(inv)
        if op=="input":
            pos,ch=divmod(a[0],8);dy,dx=divmod(pos,3)
            s=f"s{ch}[{dy-1:+d},{dx-1:+d}]"
        else:
            children=sorted([expr(a[0]),expr(a[1])])
            s=f"({children[0]} {'&' if op=='AND' else '^'} {children[1]})"
        s=("~" if inv else "")+s
        if len(s)>cap:s=s[:cap]+"... [expression truncated]"
        memo[lit]=s;return s
    return [expr(x) for x in d.outputs]

def dag_summary(d):
    reach=d.reachable();closure=core_channels(d)
    core=d.reachable([d.outputs[c] for c in closure])
    ops=lambda r:sum(d.nodes[i][0] in ("AND","XOR") for i in r)
    out={
        "binary_nodes_all_channels":ops(reach),
        "and_nodes":sum(d.nodes[i][0]=="AND" for i in reach),
        "xor_nodes":sum(d.nodes[i][0]=="XOR" for i in reach),
        "visible_recurrent_channels":closure,"visible_core_binary_nodes":ops(core),
        "relevant_local_inputs":sum(d.nodes[i][0]=="input" for i in reach),
        "output_expressions":expressions(d),
        "output_literals":d.outputs,
        "reachable_nodes":{str(i):list(d.nodes[i]) for i in reach},
    }
    return out

def hard_rollout(inputs,d,ticks=20):
    return base.rollout(inputs,lambda g:d.evaluate(base.patches(g)),ticks)

def compare_functions(a,b):
    results=[]
    for ch in range(8):
        aa,bb=project(a,[a.outputs[ch]]),project(b,[b.outputs[ch]])
        sa,sb=expressions(aa,200000)[0],expressions(bb,200000)[0]
        if sa==sb and "truncated" not in sa:
            results.append({"channel":ch,"equal":True,"method":"identical canonical expression"});continue
        variables=sorted({d.nodes[i][1] for d in (aa,bb) for i in d.reachable() if d.nodes[i][0]=="input"})
        if len(variables)>ENUM_CAP:
            results.append({"channel":ch,"equal":None,"method":"enumeration cap","relevant_input_bits":len(variables)});continue
        patch=np.zeros((2**len(variables),72),np.uint8)
        patch[:,variables]=(np.arange(len(patch),dtype=np.uint32)[:,None]>>np.arange(len(variables)))&1
        delta=aa.evaluate(patch.reshape(-1,9,8))[:,0]!=bb.evaluate(patch.reshape(-1,9,8))[:,0]
        results.append({"channel":ch,"equal":not bool(delta.any()),"method":"exhaustive relevant local assignments",
                        "relevant_input_bits":len(variables),"assignments":len(delta),
                        "disagreement_fraction":float(delta.mean())})
    return {"all_channels_equal":all(x["equal"] is True for x in results),
            "any_channel_different":any(x["equal"] is False for x in results),
            "channels":results}

def abstract_step(state,d):
    # Set codes: 1={0}, 2={1}, 3={0,1}. Exterior is {0}.
    h,w,_=state.shape
    padded=np.pad(state,((1,1),(1,1),(0,0)),constant_values=1)
    patch=np.stack([padded[y:y+h,x:x+w] for y in range(3) for x in range(3)],axis=-2).reshape(h,w,72)
    values={0:np.uint8(1)}
    complement=lambda x:((x&1)<<1)|((x&2)>>1)
    def value(lit):
        a=values[lit//2];return complement(a) if lit&1 else a
    for i in d.reachable():
        op,*args=d.nodes[i]
        if op=="input":values[i]=patch[...,args[0]]
        elif op=="AND":
            a,b=value(args[0]),value(args[1]);values[i]=((a|b)&1)|((a&b)&2)
        elif op=="XOR":
            a,b=value(args[0]),value(args[1]);values[i]=((a&b)!=0).astype(np.uint8)|(((a&complement(b))!=0).astype(np.uint8)<<1)
    return np.stack([np.broadcast_to(value(o),(h,w)) for o in d.outputs],axis=-1)

def abstract_certificate(d,size=16,ticks=64):
    state=np.full((size,size,8),3,np.uint8)
    y,x=np.indices((size,size));target=(x//2+y//2)%2
    first=None;counts=[];wrong=[];stable=None
    for t in range(ticks+1):
        known=state[...,0]!=3
        counts.append(int(known.sum()))
        wrong.append(int(np.sum(known & ((state[...,0]==2)!=target))))
        if known.all() and wrong[-1]==0 and first is None:first=t
        nxt=abstract_step(state,d)
        if np.array_equal(nxt,state):
            stable=t;break
        state=nxt
    fixed_correct=stable is not None and counts[-1]==size*size and wrong[-1]==0
    return {"size":size,"tick_budget":ticks,"known_visible_bits_by_tick":counts,
            "known_wrong_visible_bits_by_tick":wrong,
            "first_all_visible_correct":first,"abstract_fixed_point_tick":stable,
            "certifies_all_initial_states_and_future_ticks":fixed_correct}

def main():
    start=time.perf_counter()
    base.validate_algebra()
    probe_report=reconstruct(ROOT/"artifacts/reconstructed_fixed_evaluation_set.npz")
    probe=load(ROOT/"artifacts/reconstructed_fixed_evaluation_set.npz")
    inputs=probe["inputs"].astype(np.uint8);target=probe["target"]
    result={"source_commit":"02222f3ea6811108e2f14e9d8b4aea9562f1df06",
            "probe":probe_report,"enumeration_cap":ENUM_CAP,"runs":{}}
    for run in NAMES:
        print("START",run,flush=True)
        manifest=json.loads((ROOT/"results"/run/"manifest.json").read_text())
        folder=ROOT/"artifacts"/run
        exports={h:load(folder/f"final_{h}_circuit.npz") for h in ("native","common")}
        wires=[(exports["native"][f"wire_{label}_a"],exports["native"][f"wire_{label}_b"]) for label in LABELS]
        assert all(np.array_equal(exports["native"][k],exports["common"][k]) for k in exports["native"] if k.startswith("wire_"))
        assert tree_digest([x for pair in wires for x in pair])==manifest["pairing"]["wiring_sha256"]
        rep=manifest["representation"];records=[];dags={};cache={}
        for e in manifest["evaluations"]:
            update=e["update"]
            ck=load(folder/f"checkpoint_update_{update:03d}.npz")
            assert int(ck["format_version"])==2 and int(ck["update_index"])==update
            metadata=json.loads(str(ck["metadata_json"]))
            assert metadata["condition"]==manifest["condition"] and metadata["wiring_sha256"]==manifest["pairing"]["wiring_sha256"]
            q,ids=decode(ck,rep)
            saved=load(folder/f"probe_trajectory_update_{update:03d}.npz")
            assert all(np.array_equal(saved[h][0],inputs[:1]) for h in ("soft","native","common"))
            item={"update":update,"hardening":{},"recorded_soft_probe_sse":e["soft_terminal_summed_squared_error"],
                  "q_minimum_rounding_margin":min(float(abs(x-.5).min()) for x in q)}
            for hard in ("native","common"):
                gates=ids[hard];digest=tree_digest(gates)
                if digest not in cache:
                    d=base.build_dag(gates,wires)
                    full=base.rollout(inputs[:1],lambda g:base.full_step(g,gates,wires),20)
                    states=hard_rollout(inputs,d)
                    assert np.array_equal(full,states[:,:1]),(run,update,hard,"full versus simplified")
                    cache[digest]=(d,states)
                d,states=cache[digest]
                assert np.array_equal(states[:,:1],saved[hard]),(run,update,hard,"saved replay")
                metrics=base.terminal_metrics(states,target)
                assert metrics["hard_bit_errors"]==e[hard]["bit_errors"],(run,update,hard,metrics,e[hard])
                assert metrics["perfect_grid_count"]==e[hard]["perfect_grid_count"]
                if update==500:
                    assert all(np.array_equal(gates[i],exports[hard][f"gate_{label}"]) for i,label in enumerate(LABELS))
                item["hardening"][hard]={"gate_id_sha256":digest,"non_identity_gate_slots":int(sum(np.sum(g!=3) for g in gates)),
                                         **metrics,**dag_summary(d)}
                dags[(update,hard)]=d
            records.append(item)
        transitions=[]
        for u in range(0,500,50):
            a,b=dags[(u,"common")],dags[(u+50,"common")]
            transitions.append({"from_update":u,"to_update":u+50,**compare_functions(a,b)})
        final=dags[(500,"common")]
        longer=hard_rollout(inputs,final,64)
        errors_by_tick=(longer[...,0]!=target[...,0]).sum(axis=(1,2,3))
        extra={"original_probe_errors_by_tick_through_64":errors_by_tick.tolist(),
               "correct_for_all_ticks_20_through_64":bool(np.all(errors_by_tick[20:]==0)),
               "abstract_all_initial_states":abstract_certificate(final)}
        result["runs"][run]={"condition":manifest["condition"],"seed":manifest["seed"],"checkpoints":records,
                            "common_local_rule_transitions":transitions,"final_posthoc_checks":extra,
                            "final_native_vs_common":compare_functions(dags[(500,"native")],final)}
        (OUT/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(json.dumps({"run":run,"final_core":dag_summary(final),
                          "late_errors":errors_by_tick[[20,21,22,24,32,64]].tolist(),
                          "abstract":extra["abstract_all_initial_states"],
                          "seconds":time.perf_counter()-start}),flush=True)
    print("DONE",time.perf_counter()-start,flush=True)

if __name__=="__main__":main()
