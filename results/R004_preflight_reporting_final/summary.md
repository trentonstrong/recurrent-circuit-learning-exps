# R004 reporting-final preflight

Status: **passed**. This post-training preflight validates the final R004
implementation source after the aggregate reporting refinement. It does not
replace or reseed the completed formal continuations.

All four seed-0 historical replays reproduced the saved R002 update-500 state
exactly, including parameters, optimizer leaves, model/data keys, and counters.
The four-step budget comparison and ten-step uninterrupted-versus-save/resume
comparison were exact in every condition. Evaluation did not mutate live state.

All 64 parent boundaries, input-stream and wiring pairings, Boolean outputs,
saved first-probe trajectories, and FP32 soft replay checks passed. The analysis
adapter recovered all 128 update-500 structural modes and the reviewed seed-4
convergence and seed-14 phase examples. The projected 1,920-checkpoint payload
was 677,154,240 bytes; the measured formal checkpoint payload was 683,817,343
bytes.

The authoritative machine-readable evidence is `validation.json`. The original
`preflight_attempt01` failure remains recorded separately; it exposed an eager
versus compiled FP32 evaluation-form discrepancy, not a state-continuation
mismatch.
