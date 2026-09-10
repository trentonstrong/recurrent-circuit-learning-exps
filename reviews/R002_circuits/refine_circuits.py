#!/usr/bin/env python3
"""Additional exact algebraic simplification and focused dynamical checks."""
import json
from pathlib import Path
import numpy as np
import analyze_subset as a

class FactoredDAG(a.base.BooleanDAG):
    """Flatten conjunctions; remove duplicate/complement factors and absorb."""
    def __init__(self):
        super().__init__();self.factor_cache={}
    def factors(self,lit):
        if lit in self.factor_cache:return self.factor_cache[lit]
        index=lit//2
        if not lit&1 and self.nodes[index][0]=="AND":
            result=self.factors(self.nodes[index][1])|self.factors(self.nodes[index][2])
        elif lit==1:result=frozenset()
        else:result=frozenset((lit,))
        self.factor_cache[lit]=result;return result
    def raw_product(self,factors):
        value=1
        for f in sorted(factors):
            if value==1:value=f
            else:value=self.node("AND",value,f)
        return value
    def conjunction(self,x,y):
        factors=set(self.factors(x)|self.factors(y))
        while True:
            if 0 in factors or any((f^1) in factors for f in factors):return 0
            factors.discard(1)
            changed=False
            for f in tuple(factors):
                if not f&1 or self.nodes[f//2][0]!="AND":continue
                inside=set(self.factors(f^1));outside=factors-{f}
                if any((g^1) in outside for g in inside):
                    factors.remove(f);changed=True;break
                remaining=inside-outside
                if remaining!=inside:
                    if not remaining:return 0
                    factors.remove(f);factors.add(self.raw_product(remaining)^1)
                    changed=True;break
            if not changed:break
        return self.raw_product(factors)

def build(gates,wires):
    original=a.base.BooleanDAG
    try:
        a.base.BooleanDAG=FactoredDAG
        return a.base.build_dag(gates,wires)
    finally:a.base.BooleanDAG=original

def load_rule(run,update=500):
    folder=a.ROOT/"artifacts"/run
    export=a.load(folder/"final_common_circuit.npz")
    wires=[(export[f"wire_{label}_a"],export[f"wire_{label}_b"]) for label in a.LABELS]
    rep="truth" if "_truth_" in run else "categorical"
    _,ids=a.decode(a.load(folder/f"checkpoint_update_{update:03d}.npz"),rep)
    return ids["common"],wires

def validate_refinement():
    original=a.base.BooleanDAG
    try:
        a.base.BooleanDAG=FactoredDAG
        a.base.validate_algebra()
    finally:a.base.BooleanDAG=original
    # Exhaust the small algebraic patterns newly rewritten.
    rng=np.random.default_rng(44)
    patch=np.zeros((256,9,8),np.uint8)
    patch.reshape(256,72)[:,:8]=(np.arange(256)[:,None]>>np.arange(8))&1
    d=FactoredDAG();nodes=[0,1]+list(range(2,18))
    for _ in range(300):
        g=int(rng.integers(16));x,y=rng.choice(nodes,2)
        lit=d.gate(g,int(x),int(y))
        source=a.project(d,[int(x),int(y)])
        values=source.evaluate(patch)
        expected=a.base.boolean_gate(g,values[:,0],values[:,1])
        actual=a.project(d,[lit]).evaluate(patch)[:,0]
        assert np.array_equal(actual,expected)
        nodes.append(lit)
    # All nonempty abstract domains, checked against their exact set images.
    for x in (1,2,3):
        for y in (1,2,3):
            xs=[b for b in (0,1) if x&(1<<b)];ys=[b for b in (0,1) if y&(1<<b)]
            c=lambda z:((z&1)<<1)|((z&2)>>1)
            expected_and=sum(1<<b for b in {u&v for u in xs for v in ys})
            expected_xor=sum(1<<b for b in {u^v for u in xs for v in ys})
            assert ((x|y)&1)|((x&y)&2)==expected_and
            assert int(bool(x&y))|(int(bool(x&c(y)))<<1)==expected_xor

def main():
    validate_refinement()
    probe=a.load(a.ROOT/"artifacts/reconstructed_fixed_evaluation_set.npz")
    inputs=probe["inputs"].astype(np.uint8);target=probe["target"]
    result={"method":"Exact associative AND normalization, complement cancellation, and absorption; complemented AND/XOR DAG; not a minimum circuit.","runs":{}}
    for run in a.NAMES:
        gates,wires=load_rule(run)
        old=a.base.build_dag(gates,wires);d=build(gates,wires)
        old_states=a.hard_rollout(inputs,old,64);states=a.hard_rollout(inputs,d,64)
        assert np.array_equal(old_states,states),(run,"refinement mismatch")
        per_cell=states[...,0]!=target[...,0]
        abstract=np.full((16,16,8),3,np.uint8)
        for _ in range(64):abstract=a.abstract_step(abstract,d)
        wrong=((abstract[...,0]!=3)&((abstract[...,0]==2)!=target[0,...,0]))
        item={"basic":a.dag_summary(old),"refined":a.dag_summary(d),
              "abstract_certificate":a.abstract_certificate(d),
              "provably_wrong_cells_after_tick_64":np.argwhere(wrong).tolist(),
              "all_probe_visible_period2_from_tick20":bool(np.array_equal(states[20:-2,...,0],states[22:,...,0])),
              "all_probe_full_state_period2_from_tick20":bool(np.array_equal(states[20:-2],states[22:])),
              "full_state_fixed_from_tick20":bool(np.array_equal(states[20:-1],states[21:])),
              "all_probes_same_visible_at_tick20":bool(np.all(states[20,...,0]==states[20,0,...,0])),
              "all_probes_same_visible_at_tick21":bool(np.all(states[21,...,0]==states[21,0,...,0])),
              "visible_wrong_pixels_tick20_first_probe":np.argwhere(per_cell[20,0]).tolist(),
              "original_probe_errors_by_tick":per_cell.sum(axis=(1,2,3)).tolist(),
              "first_probe_thresholded_soft_errors":{}}
        for update in (250,400,450,500):
            saved=a.load(a.ROOT/"artifacts"/run/f"probe_trajectory_update_{update:03d}.npz")
            item["first_probe_thresholded_soft_errors"][str(update)]={
                "thresholded_soft_errors":int(np.sum((saved["soft"][-1,0,...,0]>=.5)!=target[0,...,0])),
                "common_hard_errors":int(np.sum(saved["common"][-1,0,...,0]!=target[0,...,0])),
                "soft_sse":float(np.sum((saved["soft"][-1,0,...,0]-target[0,...,0])**2))}
        if run.endswith("seed04_categorical_no_decay") or run.endswith("seed04_categorical_reference_decay"):
            def intended(g):
                p=a.base.patches(g)
                out=g.copy()
                out[...,0]=p[...,2,6]|(1^(p[...,0,6]|p[...,8,6]))
                out[...,6]=p[...,2,0]
                return out
            # Exhaust the four relevant inputs to both output equations.
            relevant=[0*8+6,2*8+6,8*8+6,2*8+0]
            patch=np.zeros((16,9,8),np.uint8)
            patch.reshape(16,72)[:,relevant]=(np.arange(16)[:,None]>>np.arange(4))&1
            observed=d.evaluate(patch)
            expected_v=patch[:,2,6]|(1^(patch[:,0,6]|patch[:,8,6]))
            assert np.array_equal(observed[:,0],expected_v)
            assert np.array_equal(observed[:,6],patch[:,2,0])
            item["two_field_equations_exhaustively_verified"]=True
        result["runs"][run]=item
        a.OUT.joinpath("refined_analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        print(run, "basic/refined binary",item["basic"]["binary_nodes_all_channels"],item["refined"]["binary_nodes_all_channels"],
              "core",item["refined"]["visible_core_binary_nodes"],"channels",item["refined"]["visible_recurrent_channels"],
              "period2",item["all_probe_full_state_period2_from_tick20"],
              "abstract",item["abstract_certificate"]["certifies_all_initial_states_and_future_ticks"],flush=True)
    # Compare the two seed-4 cores directly for arbitrary local inputs.
    left=build(*load_rule(a.NAMES[0]));right=build(*load_rule(a.NAMES[2]))
    result["seed04_cross_condition_local_comparison"]=a.compare_functions(left,right)
    a.OUT.joinpath("refined_analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":main()
