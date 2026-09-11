# R003 — Changing the categorical representative of a fixed relaxed gate

Status: completed, 2026-09-11. The focused preflight and fixed 96-checkpoint
diagnostic passed. The authoritative v1 protocol is the
[implementation handoff](HANDOFF.md) and its
[machine-readable configuration](../../configs/experiments/r003_same_function_v1.json).
The [original baseline handoff](../../docs/RECURRENT_BASELINE_HANDOFF.md) supplies
the earlier derivation; the R003 handoff fixes the previously open choices.

The canonical result is
[`R003_formal_attempt02`](../../results/R003_formal_attempt02/summary.md). The
same-function controls passed for all 288 arms. Changing the representative
changed the induced q response and the visible-output response in every primary
checkpoint contrast, with the smaller independent step supporting the local
linear prediction. These are post-training mechanism measurements, not evidence
of improved training or a causal explanation of circuit size.

Measure whether replacing a categorical mixture at fixed effective truth table
changes the induced logit-SGD response, and whether that change reaches the
visible outputs of the recurrent circuit. The existence of such coordinate
dependence is an algebraic possibility; its magnitude and observable effects
at actual checkpoints are the measurements.

Use all 32 categorical R002 `formal_attempt01` runs (seeds 0–15 under both decay
histories), at saved updates 0, 250, and 500: 96 logical checkpoint cases.
Evaluate in FP64 on fixed saved probe arrays, with a declared numerical
eligibility mask. Compare the original, halfway, and factorized mixtures. Each
arm receives independent one-step SGD diagnostics at eta 0.001 and 0.0001,
with no decay, cost, clipping, momentum, or use of checkpoint Adam moments.

For a categorical gate distribution p with effective truth entries q, construct
the factorized representative

\[
p'_g=\prod_{r=1}^{4}q_r^{T_{gr}}(1-q_r)^{1-T_{gr}}.
\]

For interior q this is a positive distribution with the same effective table.
Interpolate in probability space with `p_alpha=(1-alpha)*p+alpha*p_factorized`.
Materialize the replaced logits as independent leaves before differentiating;
do not backpropagate through the replacement or constrain subsequent SGD to the
factorized family. This is not the exact ensemble of recurrent circuit
executions. Follow the handoff's normalization, boundary, and rounding policies.

Freeze wiring, probe arrays, recurrence, and the observer. Verify q, full soft
state trajectories, loss, state JVPs, and q gradients before interpreting each
comparison. Compare local Jacobians, induced SGD matrices, q/output responses,
and actual finite steps. Record native argmax changes separately; common table
rounding is unchanged by exact same-q movement.

This diagnostic does not establish that redundant coordinates help, that SGD
naturally explores exact fibers, or that the intervention causes smaller final
circuits. It is separate from longer training and any Adam continuation. Retain
all original artifacts and record every failed or missing case. Implement the
focused preflight before executing the fixed cohort.
