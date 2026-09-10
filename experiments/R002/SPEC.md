# R002 — Matched gate coordinates and weight decay

Status: proposed; implementation handoff prepared, full sweep not launched.
The [implementation handoff](IMPLEMENTATION_HANDOFF.md) specifies the next local
Codex task: direct notebook parity, paired runner, and bounded validation.
The [full handoff](../../docs/RECURRENT_BASELINE_HANDOFF.md) is the detailed protocol.

Use 16 paired seeds, 0–15, with the synchronous architecture and 500 updates:

| Condition | Gate coordinates | AdamW weight decay |
| --- | --- | ---: |
| categorical_reference_decay | 16-way softmax | .01 |
| truth_reference_decay | Four sigmoid truth entries | .01 |
| categorical_no_decay | 16-way softmax | 0 |
| truth_no_decay | Four sigmoid truth entries | 0 |

All trials start from scratch. Pair actual wiring and training arrays, hold the
reference schedule, .05 learning rate, clipping, loss, precision, and evaluation
set fixed. Initialize each sigmoid table from the categorical effective table
using `logit(q)`. Verify matched initial soft rollouts and state JVPs.

Report both native extraction and common truth rounding, using `q >= .5` for a
true entry. Native categorical ties follow the source's argmax convention.
Equal parameter decay or learning rate does not mean equal regularization or
functional step size. Record clipping and effective-table displacement to make
that distinction observable. No extra noise or description cost belongs here.

Report per-seed outcomes, paired comparisons, uncertainty over training seeds,
soft/hard gap, first exact evaluation checkpoint, final results, compile/runtime,
and memory. Probe grids are not independent training trials. This experiment
compares the specified optimization recipes; learning-rate matching or sweeps
are separately declared experiments.

Choose concurrency after profiling; preserve each trial's arrays and recipe.
Do not launch this sweep merely because an implementation compiles.
