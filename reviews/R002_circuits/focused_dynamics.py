#!/usr/bin/env python3
"""Focused, exact runtime and neutral-snapshot checks. No optimizer updates."""
import json
import numpy as np
import analyze_subset as a
import refine_circuits as r

def main():
    extra={}
    probe=a.load(a.ROOT/"artifacts/reconstructed_fixed_evaluation_set.npz")
    target=probe["target"][0,...,0]
    for run in a.NAMES:
        d=r.build(*r.load_rule(run))
        fixed=np.stack([np.zeros((16,16,8),np.uint8),np.ones((16,16,8),np.uint8)])
        states=a.hard_rollout(fixed,d,64)
        errors=(states[...,0]!=target).sum(axis=(2,3))
        item={"zero_and_one_initialization_errors_by_tick":errors.tolist()}
        if "_seed04_" in run:
            core=a.project(d,[d.outputs[0],d.outputs[6]])
            used={core.nodes[i][1] for i in core.reachable() if core.nodes[i][0]=="input"}
            assert used=={6,22,70,16}
            item["exact_core_input_support"] = sorted(used)
            spans=[]
            for u in (250,300,350,400,450):
                ga,_=r.load_rule(run,u);gb,_=r.load_rule(run,u+50)
                da=r.build(*r.load_rule(run,u));db=r.build(*r.load_rule(run,u+50))
                spans.append({
                    "from_update":u,"to_update":u+50,
                    "common_gate_slot_changes":sum(int(np.sum(x!=y)) for x,y in zip(ga,gb)),
                    "all_channels_equal":a.compare_functions(da,db)["all_channels_equal"],
                    "same_visible_core":a.expressions(a.project(da,[da.outputs[0],da.outputs[6]]))==a.expressions(a.project(db,[db.outputs[0],db.outputs[6]])),
                })
            item["late_transitions"]=spans
        if "_seed14_" in run:
            states=a.hard_rollout(probe["inputs"].astype(np.uint8),d,24)
            assert np.array_equal(states[20],states[22])
            assert not np.array_equal(states[20],states[21])
            assert np.all(states[20,...,0]==target)
            witness=states[21,0].copy()
            replay=a.hard_rollout(witness[None],d,20)
            assert np.array_equal(replay[0],replay[2])
            assert np.array_equal(replay[0],replay[20])
            nerrors=int(np.sum(replay[-1,0,...,0]!=target))
            assert nerrors==128
            witness_doc={
                "run_id":run,"checkpoint_update":500,"hardening":"common",
                "construction":"Full eight-channel state at runtime tick 21 of original probe 0.",
                "array_shape":[16,16,8],"array_dtype":"uint8",
                "codec":"C-order flattened Boolean bits, numpy.packbits bitorder=big, bytes encoded as lowercase hex",
                "initial_state_packed_hex":np.packbits(witness.reshape(-1),bitorder="big").tobytes().hex(),
                "initial_state_array_sha256":a.tree_digest([witness]),
                "visible_errors_at_tick_20":nerrors,
                "full_state_repeats_after_two_ticks":True,
                "conclusion":"This valid initial state fails every even readout tick, including the trained tick 20.",
            }
            decoded=np.unpackbits(np.frombuffer(bytes.fromhex(witness_doc["initial_state_packed_hex"]),dtype=np.uint8),bitorder="big").reshape(16,16,8)
            assert np.array_equal(decoded,witness)
            (a.OUT/"seed14_phase_counterexample.json").write_text(json.dumps(witness_doc,indent=2)+"\n")
            item["phase_counterexample"]=witness_doc
            item["original_probe_phase_certificate"]={
                "all_32_full_states_at_20_equal_their_states_at_22":True,
                "all_32_visible_states_at_20_correct":True,
                "all_32_visible_states_at_21_identical":bool(np.all(states[21,...,0]==states[21,0,...,0])),
                "all_32_visible_errors_at_21":int(np.sum(states[21,...,0]!=target)),
                "infinite_periodicity_for_these_initial_states":"Determinism and exact full-state equality at ticks 20 and 22 imply repetition forever.",
            }
        if "_seed08_" in run:
            state=np.full((16,16,8),3,np.uint8)
            for tick in range(65):
                wrong=(state[...,0]!=3)&((state[...,0]==2)!=target)
                if wrong.any():
                    item["first_universally_wrong_cells"]={"tick":tick,"yx_coordinates":np.argwhere(wrong).tolist()}
                    break
                state=a.abstract_step(state,d)
            assert item["first_universally_wrong_cells"]["tick"]<=20
        extra[run]=item
    (a.OUT/"focused_dynamics.json").write_text(json.dumps(extra,indent=2)+"\n")
    print(json.dumps({"phase_counterexample":extra[a.NAMES[-1]]["phase_counterexample"],"seed08_wrong_cells":extra[a.NAMES[-2]]["first_universally_wrong_cells"]},indent=2))

if __name__=="__main__":main()
