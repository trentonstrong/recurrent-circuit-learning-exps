# R001 — Synchronous recurrent positive control

Status: canonical seed-23 run completed on 2026-09-09. See the
[result summary](../../results/R001_20260909T231016Z_seed23_jaxgpu_attempt01/summary.md).

## Question and fixed recipe

Does the audited synchronous checkerboard learner train in the recorded runtime?
Use the [machine-readable specification](../../configs/experiments/r001_sync_reference.json).
Its reference recipe is copied from the immutable
[source audit](../../references/difflogic_ca/source_manifest.json).

| Setting | Value |
| --- | --- |
| Seed; optimizer updates | 23; 500 |
| Grid; channels; batch | 16 x 16; 8; 2 |
| Runtime | 20 synchronous ticks |
| Target; exterior | Checker squares of width 2; constant zero padding |
| Gates; wiring | 16-function softmax; fixed source wiring |
| Initialization | Gate A logit 10; other logits 0; fresh Boolean grid each update |
| Loss | Channel-0 terminal squared error, summed over batch and grid |
| Optimizer | AdamW .05 learning rate, betas (.9,.99), weight decay .01 |
| Clipping | Elementwise at magnitude 100, before optimizer |
| Precision | FP32 training, with backend flags recorded |

Preserve source architecture, target construction, RNG semantics, and native
argmax hardening. Resolve and record optimizer defaults through R000. Keep the
published evaluation/logging distinguishable from the project's added probes.
Cross-entropy, extra noise, learned wiring, emit logic, and description pressure
are separate interventions, not part of this run.

## Checkpoints and added evaluation

Save initial state and states after updates 50,100,...,500, including parameters,
optimizer moments, RNG, update index, and enough metadata to resume. Log cheap
loss and gradient summaries every update. Distinguish pre-update training loss
from evaluation of the final parameters after update 500.

Before training, generate 32 Boolean initial grids from a separate JAX PRNG stream
seeded with 1000023. Save and hash the arrays. Use the source's target convention
and reuse this set at every checkpoint. It must not advance the training RNG or
affect training, tuning, or checkpoint selection.

At each checkpoint report terminal soft squared error, native hard bit error,
fraction of grids reconstructed perfectly, and selected visible/hidden runtime
trajectories. Define the soft/hard gap using the same dataset and loss reduction.
The primary checkpoint is update 500. First exact checkpoint is descriptive,
not a rule for selecting the model. Exact reconstruction on the added finite
set means zero hard errors on all 32 grids; the paper does not guarantee that
additional criterion.

## Deliverables and interpretation

Record source and code pins, config and array hashes, environment locks, actual
hardware and driver, precision/flags, run status, timings, peak memory, and artifact
locations in the [result ledger](../../results/README.md). Export the final Boolean
circuit. Preserve unsuccessful and interrupted runs; do not rerun a seed until
it works and report only the survivor.

One run is a working-example check, not a success-rate estimate. Report the
source-compatible results and added probe results separately. If the reference
fails, examine parity, implementation, and runtime evidence before attributing
failure to recurrent logic learning. Any retuning has a new protocol identity.

Review this result before launching [R002](../R002/SPEC.md). Longer runtimes,
larger grids, perturbations, asynchronous updates, and counter generation require
separately specified evaluations or experiments.
