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

## D008 — Use uv for experiment environments

The owner confirmed uv on 2026-09-09. Use independent uv projects for the
historical JAX CPU oracle and modern GPU runtime, each with a validated lock and
exact interpreter pin. Record uv itself in run provenance and use locked mode
for reproduction. R000 will resolve and validate these environments locally;
this decision does not constitute a completed dependency lock.

## D009 — Separate longer training from frozen hardening probes

On 2026-09-12, retain the known recurrent loss and optimizer while studying why
some relaxed models fail gate hardening. R004's finite-budget gap is not proof
of convergence or of an unavoidable analog computation. R006 continues all
64 original trajectories from update 2,000 to a fixed 10,000 endpoint, with
no outcome-based selection or early stopping. The endpoint is a fivefold
total budget relative to R004, adding 512,000 updates across the same cohort.

At five fixed checkpoints, evaluate Q-alpha = Q + alpha*(H(Q)-Q) directly in
truth-table coordinates on 33 fixed alphas and the existing 66-start probe set.
Keep the entire curve, including failures followed by recoveries. Record actual
rounding displacement and numerical sensitivity; equal alpha need not mean
equal parameter movement across checkpoints. These frozen evaluations cannot
change live parameters, optimizer state, random streams, or the training loss.
The path changes the relaxed function while preserving its common-rounded
circuit, and is distinct from R005's same-function neutral movement.

Keep evaluation every 50 updates and store resumable checkpoints every 250
to limit artifacts; retain every update record outside git with hashes.
This is a storage choice, not altered training mathematics. Keep R005's
protocol and source selection unchanged. The
[R006 handoff](../experiments/R006/HANDOFF.md) and
[configuration](../configs/experiments/r006_training_and_hardening_v1.json)
define the preflight, paired endpoints, diagnostics, and validation boundary.
