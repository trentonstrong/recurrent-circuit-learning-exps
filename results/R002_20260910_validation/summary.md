# R002 paired-runner validation

Status: passed on 2026-09-10. The runner is implemented and ready for review;
the formal 64-run sweep has not been launched.

## Coordinate and kernel checks

Both representations use one declared multilinear four-entry truth-table kernel.
Categorical effective tables use `softmax(z) @ T`; this dot explicitly requests
JAX `Precision.HIGHEST`. Direct truth coordinates use `sigmoid(z)`. Initial truth
logits are computed as `logit(q)` in FP32 without clamping.

The retained first GPU kernel report failed because the default 16-by-4 matrix
multiply used reduced precision: the gate bridge differed by `6.2018633e-5` and
the initial 20-tick trajectory by `0.04832089`. This was a runtime precision
adaptation, not a scientific recipe change. With highest dot precision, the gate
bridge error is `5.9604645e-8`. The largest source-to-common errors in the passing
report are `1.9788742e-5` for an initial trajectory, `6.9737434e-6` for gradients,
and `2.0027161e-5` after one AdamW update. Initial categorical-to-truth trajectories
differ by at most `1.5854836e-5`; their state JVPs match exactly. All are within
the existing FP32 policy.

Independent historical-CPU FP64 central differences pass for both coordinate
systems. Maximum gradient errors are `3.6113213e-12` categorical and
`1.8913829e-12` truth. Native categorical argmax and common truth rounding exports
all match independent hard execution exactly. Truth native hardening is the same
four-threshold rule as common hardening; it is never decoded with a four-way
argmax.

## Seed-23 development preflight

Each condition ran in a fresh GPU process for two full-shape updates: batch 2,
16-by-16 grids, eight channels, and 20 synchronous ticks. All four conditions
share wiring SHA-256
`ab4065cbd4ebb853b822b624943c532ea8e1a638d93c7b2aed498180a7da0268`
and ordered two-batch stream SHA-256
`d15fe5ab4027406f62f5ca54172f6cde738728a7f32292c05e7ff86ad191dc93`.
Attempt-1 reports are retained but superseded: attempt 2 additionally regenerates
the second batch from the checkpoint-restored data key.

| Condition | Compile s | First / second update ms | Peak bytes | Max initial q discrepancy |
| --- | ---: | ---: | ---: | ---: |
| categorical_reference_decay | 8.962 | 119.7 / 82.1 | 1,681,886,208 | 0 |
| truth_reference_decay | 8.332 | 119.8 / 81.8 | 1,679,096,320 | `5.9604645e-8` |
| categorical_no_decay | 8.883 | 119.6 / 82.0 | 1,681,886,208 | 0 |
| truth_no_decay | 8.364 | 120.4 / 82.1 | 1,679,097,600 | `5.9604645e-8` |

Every checkpoint-after-update-1 regenerated the exact next batch and data key,
then resumed update 2 with exact parameters, optimizer state, model key, loss,
and update index. Compilation used cloned state;
the measured run restarted from its original state. No first-update gradient
entry exceeded the elementwise clip bound of 100. As expected, coordinate systems
take different optimizer steps immediately. The development hard result remained
at 256 errors of 512 visible bits after two updates; it is not a learning-success
test and is excluded from the formal cohort.

Single-process concurrency is selected for the first sweep. Live peak allocations
are small relative to the device, but separate default JAX processes can compete
for allocator reservation; concurrency one preserves the path actually validated.

## Formal launch command

The frozen configuration SHA-256 is
`9632f2135e8262903c40cdaeba211e4f5bebac22c2b8f45c0e2100b993111bb9`.
After review, the exact launch command is:

```sh
JAX_ENABLE_X64=0 uv run --project envs/jax-gpu --locked python scripts/r002.py sweep --config configs/experiments/r002_paired_gate_coordinates.json --concurrency 1 --sweep-id formal_attempt01
```

The runner records per-update gradient/clipping, parameter and effective-table
displacement, native/common gate changes, saved-checkpoint probe metrics,
layer/readout-cone entropy and rounding margins, selected trajectories, full
checkpoint state, pairing hashes, timing, memory, and final native/common circuit
exports. Simplified Boolean-node counts remain a post-hoc analysis of completed
hard circuits and do not alter trial execution.
