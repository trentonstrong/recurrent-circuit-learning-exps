# R002 — Matched gate coordinates and weight decay

Status: formal 64-run sweep completed in
[`formal_attempt01`](../../results/R002_formal_attempt01/summary.md).
The [implementation handoff](IMPLEMENTATION_HANDOFF.md) specified direct notebook
parity, the paired runner, and bounded validation. Those checks are recorded in
the [R000 parity result](../../results/R000_20260910_direct_notebook_parity_attempt01/summary.md)
and [R002 validation result](../../results/R002_20260910_validation/summary.md).
The [full handoff](../../docs/RECURRENT_BASELINE_HANDOFF.md) is the detailed protocol.
The paired cohort found 1/16 exact categorical/reference-decay trials, 1/16
exact truth/reference-decay trials, 2/16 exact categorical/no-decay trials, and
0/16 exact truth/no-decay trials. Paired uncertainty does not support an
advantage for either coordinate system or decay setting. Its checkpoint,
trajectory, and final-circuit payloads are published under the experiment-scoped
[`experiment/R002` release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R002).

The [selected circuit review](../../reviews/R002_circuits/REVIEW.md) is complete.
The subsequent [runtime-profile handoff](RUNTIME_PROFILE_HANDOFF.md) specifies an
agreed posthoc diagnostic across all 64 frozen final circuits. That profile is
ready for implementation; it has not run and does not revise this experiment's
20-tick readout or original success scores.

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
