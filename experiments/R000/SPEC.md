# R000 — Source capture and implementation parity

Status: completed, including direct vendored-notebook execution and
X64-disabled sampling parity. See the
[initial results](../../results/R000_20260909_jax_parity/summary.md),
[direct parity result](../../results/R000_20260910_direct_notebook_parity_attempt01/summary.md),
and [follow-up implementation handoff](../R002/IMPLEMENTATION_HANDOFF.md).
The original scope below is retained. Read [AGENTS.md](../../AGENTS.md) and
[workstation setup](../../docs/WORKSTATION_SETUP.md) before changes.

## Question

Can a standalone implementation reproduce the pinned notebook's mathematics,
and can the selected office GPU runtime reproduce reference numerical fixtures?

## Work

1. Fetch the [pinned source](../../references/difflogic_ca/source_manifest.json),
   verify its byte-level SHA-256, and preserve Apache-2.0 attribution for vendored
   or extracted code. Extract the synchronous recipe without notebook globals or
   execution-order dependencies. Inventory any modern API changes.
2. Establish a JAX/JAXLIB 0.4.33 CPU oracle with a resolved dependency lock.
   Establish a separate modern JAX GPU environment on the actual 5090. Inspect
   the installed Optax defaults, including epsilon semantics, instead of assuming
   another optimizer API is equivalent. Record both environment locks.
3. Export deterministic fixtures: all wiring arrays, initial and nontrivial gate
   logits, Boolean and fractional inputs, targets, and RNG states. Preserve the
   source's key splitting and reference seed behavior; compare actual arrays.
4. Check the 16 gates, fractional extensions, gate A orientation, patch order,
   zero boundaries, kernel/channel sharing, flattening, and intermediate shapes.
   Compare one update, the 20-tick rollout, summed terminal loss, parameter
   gradients, and a clipped AdamW update between reference and modern JAX.
5. Provide hard Boolean execution/export and a focused checkpoint-resume check.
   Hard gate tables and Boolean executions must agree exactly. The checkpoint
   restores model, optimizer, RNG, and update index; a resumed next update must
   agree with uninterrupted execution within the declared numerical contract.
6. If implementing PyTorch, run the same checks against the JAX fixture arrays.
   Establish the JAX runner first. Record a port as a port; do not use the legacy
   counter learner as the implementation of this reference.
7. Profile one compiled forward/backward update at the unchanged R001 shape.
   Record compile time, synchronized runtime, and memory. Reset after profiling
   so warm-up cannot alter the canonical initialization or random stream.

## Acceptance and artifacts

Use exact equality for wiring, gate tables, and Boolean execution. Start local
FP64 probes with `atol=1e-10, rtol=1e-8` and FP32 probes with
`atol=1e-6, rtol=1e-4`. Enable actual FP64 in the oracle when requested. These
are starting probe tolerances, not permission to hide deep-rollout discrepancies.
Report maximum errors and the earliest differing layer/tick; justify any changed
tolerances using the observed numerical behavior.

Test both reference near-identity initialization and nontrivial fixed logits.
An identity-only check can miss orientation and wiring mistakes. Compare gradients
within a parameterization across frameworks, not categorical and truth-coordinate
parameter gradients against each other.

Commit implementation, focused checks, exact environment locks, and a parity
report under `results/<run_id>/`. Save fixtures and checksums with provenance.
Make a concrete R001 command available in that report and the README. If a check
fails, preserve the failure and resolve its cause before interpreting training.
No broad seed or hyperparameter sweep belongs to R000.

After acceptance, proceed to [R001](../R001/SPEC.md). The
[full handoff](../../docs/RECURRENT_BASELINE_HANDOFF.md) supplies the detailed
source audit and dimensionality explanation.
