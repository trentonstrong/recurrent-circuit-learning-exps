# R006 — Longer training with frozen hardening probes

Status: specified for implementation, 2026-09-12. No R006 runner, preflight,
training, or GPU benchmark has run as part of this handoff.

The fixed v1 protocol is the [implementation handoff](HANDOFF.md) and
[configuration](../../configs/experiments/r006_training_and_hardening_v1.json).
[Source identities](SOURCE_PINS.json) pin all 64 R004 update-2,000 parents and the
reference implementation at c7680962270b0cd2a1ca22f46a4d7dbc7f0ba72e.

Continue every original seed/condition trajectory from update 2,000 to 10,000:
8,000 additional updates each, 512,000 total. Preserve the FP32 AdamW recipe,
constant learning rate, clipping, batch size, fresh-input stream, wiring,
parameters, moments, counters, keys, 20-tick rollout, zero exterior, and summed
terminal channel-0 squared error. The only training intervention is more
updates, including the recipe's continued weight decay where applicable.

At updates 2,000, 4,000, 6,000, 8,000, and 10,000, freeze each model and evaluate

\[
Q_\alpha=(1-\alpha)Q+\alpha H(Q),\qquad H(Q)=\mathbf1[Q\geq1/2],
\]

over a fixed 33-point alpha grid and the same 66 declared initial states.
Q contains the effective four-entry truth tables, materialized in FP32.
The path is evaluated directly through the common multilinear kernel, with
the original arrays at alpha 0 and exactly Boolean tables at alpha 1.
All gates remain in their original common-rounding region along this path.
No interpolated model enters training, and no runtime state is thresholded
before the observer. This is distinct from R005's same-Q neutral intervention.

The primary endpoint remains all-32 original-probe common-hard correctness at
runtime tick 20, now at optimizer update 10,000. Preserve ordinary evaluations
every 50 updates; store full resumable checkpoints every 250. The latter is a
declared storage-cadence change, with no change to the update mathematics.
Measure constructive circuit size every 500 updates and the existing
66-start/256-tick hard runtime profile at 2,000 and 10,000.

Record complete hardening curves, thresholded soft outputs, truth-table
displacement, and per-tick state divergence from the unmodified relaxed model.
Do not assume monotonic deterioration along alpha. A run that fits its relaxed
outputs but fails hardening may still be undertrained; neither a final failure
nor a sampled hardening curve proves convergence or an unavoidable analog
computation. The leaky-memory example and interpretation limits are in the
handoff.

Primary inference compares update-2,000 and update-10,000 exactness within each
condition, with paired tests and Holm correction over four conditions.
Secondary curves, coordinate comparisons, size trajectories, and baseline
failure subgroups are descriptive, with paired seed-level summaries. Repeated
checkpoints, alphas, gates, and initial grids are not independent trials.

R005 remains unchanged and uses its original R002 sources. Implement and validate
R006 on the workstation before launching its full continuation. This document
does not claim an execution result.
