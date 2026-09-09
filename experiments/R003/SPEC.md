# R003 — Movement within a gate's neutral fiber

Status: proposed mechanism study after recurrent reproduction. Select checkpoints,
probe arrays, precision, and diagnostic step size before execution. The
[full handoff](../../docs/RECURRENT_BASELINE_HANDOFF.md) contains the derivation.

For a categorical gate distribution p with effective truth entries q, construct
the factorized representative

\[
p'_g=\prod_{r=1}^{4}q_r^{T_{gr}}(1-q_r)^{1-T_{gr}}.
\]

For interior q this is a positive distribution with the same effective table.
This is not the exact ensemble of recurrent circuit executions. Verify
normalization, positivity, and table equality numerically; finite precision
near a boundary requires explicit handling, not hidden clipping.

Freeze the other parameters and probe arrays. On cloned checkpoints, exclude
cost and decay and verify that soft states, loss, and state JVPs remain unchanged.
Record any native argmax changes. Compare the gate-coordinate Jacobians, their
induced SGD matrices, and a single declared SGD step.

Changing the representative can change the optimizer's subsequent functional
movement despite an unchanged current soft function. The diagnostic tests that
mechanism, not a presumption that redundant coordinates help. Extending it to
Adam requires a declared policy for first/second moments and coordinate changes.
Do not overwrite the original training trajectories.
