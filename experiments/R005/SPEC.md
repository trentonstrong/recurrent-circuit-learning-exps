# R005 — Stochastic exploration within same-function fibers

Status: specified for implementation, 2026-09-12. The experiment runner and GPU
intervention have not been implemented or executed by this handoff. The fixed
v1 protocol is the [handoff](HANDOFF.md) and
[configuration](../../configs/experiments/r005_stochastic_neutral_exploration_v1.json).

R003 established that the representative of a categorical gate mixture can
change its local SGD response while preserving its effective truth table.
R005 tests a constructive stochastic exploration rule: make small, feasible
neutral probability exchanges, favoring exchanges predicted to rotate the next
response, while retaining a uniform component.

Use the same 96 original R002 checkpoint cases as R003: both categorical
training histories, seeds 0–15, updates 0, 250, and 500. Every diagnostic clone
starts from its untouched source. Retain the FP64, plain-logit-SGD mechanism
setting and the fixed two-input loss/four-input observation convention. The
inherited loss is summed terminal squared error at runtime tick 20.

For each checkpoint, compare identity, 16 uniform draws, and 16
geometry-weighted draws. Also derive 16 pairs with equal per-gate probability
displacement from those same draws. Freeze rho = 0.1 and gamma = 0.75 for the
weighted rule; these are initial design choices, not tuned optima. Apply one
exchange to each eligible shared gate. Take independent single SGD steps at
eta = 0.001 and 0.0001 from each representative. This accounts for 6,240
representatives and 12,480 diagnostic steps, before failures or verified reuse.

The primary endpoint is the change in stochastic dispersion of normalized
visible-output responses at update 250, comparing the amplitude-matched
weighted and uniform arms, separately for the two training histories. Use
observation inputs 2 and 3, which do not enter the loss. Measure average drift
separately: a stochastic intervention need not produce a broader distribution.
Report updates 0 and 500, raw proposal laws, response norms, finite-step effects,
truth-table correlators, and extraction changes as secondary diagnostics.

Implement the operation as batched JAX array calculations within a compiled GPU
update. Reuse the recurrent q-gradient after a neutral move and recompute its
mapping to logit gradients; validate against fresh differentiation. Benchmark
the three raw arms separately, with evolving parameters inside timed blocks
and no CPU tensor round trip. FP64 benchmark results do not predict FP32 speed.

This is a checkpoint mechanism experiment and implementation-cost measurement.
It does not continue R004, reset its optimizer, select a better training run,
establish a stationary sampling law, or demonstrate improved learning or shorter
complete descriptions. Those training questions follow after this rule is
validated. No new recurrent training cohort is launched by this specification.
