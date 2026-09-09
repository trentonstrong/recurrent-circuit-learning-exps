# Decisions

These entries record the handoff accepted for implementation on 2026-09-09.
They are not claims that the proposed experiments have run.

## D001 — Establish recurrent learning before the counter adaptation

Use the official synchronous checkerboard experiment as the first positive
control. Its recurrent rollout is directly relevant to training dynamics, while
one-step Game of Life supervision would leave that issue unresolved. A successful
checkerboard run will not establish sequence generation or compression.
Evidence: [audited source](../references/difflogic_ca/source_manifest.json) and
[handoff](RECURRENT_BASELINE_HANDOFF.md).

## D002 — Separate gate coordinates from the full architecture

Compare 16-way categorical gates and four sigmoid truth entries on the same
working recurrent task. Match actual wiring, data, and initial effective gates.
Cross with weight decay .01 versus zero because decay is coordinate-dependent.
Keep native hardening and common truth-table rounding separate. The first
comparison measures the specified training recipes; it is not an equalized
functional step-size comparison. Details: [R002](../experiments/R002/SPEC.md).

## D003 — Preserve independent trials and complete descriptions

No trained weights carry into primary independent runs. Charge the declared
machine, initialization, observation, runtime contract, and exceptions when
reporting a description score. Parameter counts remain labeled proxies.
An exceptions codec is allowed; infinite generalization is a separate claim.

## D004 — Treat the 5090 runtime as an explicit adaptation

Retain JAX/JAXLIB 0.4.33 as the notebook's historical version pin. Resolve a
compatible CPU environment for reference fixtures, and validate a modern
Blackwell-capable GPU environment against them. The notebook has no complete
dependency lock, so even the CPU profile is a reconstructed reference environment.
Do not claim the new machine reproduces the original hardware's bitwise trajectory.
The source manifest remains unchanged. Evidence and open items:
[workstation setup](WORKSTATION_SETUP.md).

## D005 — Freeze the added evaluation independently of training

R001 retains the source's seed 23 and 500-update budget. The added 32-grid probe
set uses a separate JAX PRNG stream seeded with 1000023, is saved and hashed before
training, and never advances the training RNG. Evaluate initial and every-50-update
checkpoints through update 500. This probe set is a project addition, not a result
or guarantee from the paper. The final update is the primary checkpoint; the
probe set must not select a checkpoint or trigger tuning.

## D006 — Profile one GPU process before choosing concurrency

Keep the reference batch, grid, and unroll unchanged. Measure compilation,
steady-state update time, and peak memory first. The number of CPU threads is not
the number of viable simultaneous GPU trials. Any scheduling changes retain
the same scientific recipe and recorded per-trial randomness.

## D007 — Target native Arch Linux

The owner confirmed native Arch Linux on 2026-09-09. Update the hardware profile
and setup guidance accordingly; driver and kernel versions still require local
inspection. Pin experiment Python interpreters independently of the system
interpreter and keep the historical CPU and modern GPU environments separate.
Record host changes alongside dependency locks so later runtime differences are
observable. This changes environment guidance, not the R001 scientific recipe.
