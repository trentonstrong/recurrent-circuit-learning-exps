# R004 formal continuation: formal_attempt01

Status: **completed**. All 64 scheduled R002 descendants reached global update
2,000 from their own verified update-500 parameters, AdamW state, RNG keys, and
wiring. This is the same 16 paired seed groups under four continued conditions,
not 64 new trials.

The cohort contains all 96,000 requested updates, 1,920 new checkpoints, 1,920
new probe trajectories, and 768 new Boolean-circuit exports. For every seed,
the four condition-specific continuation batch histories have the same complete
indexed digest. The failure ledger is empty. An independent closeout audit
recomputed all batch digests and verified every recorded checkpoint,
trajectory, circuit, result, and runtime-array byte length and SHA-256.

The artifact payload is 976,096,916 bytes: 683,817,343 checkpoint bytes,
274,738,539 trajectory bytes, and 17,541,034 circuit-export bytes. Aggregate
synchronized update time was 7,772.55 seconds; summed per-run training-loop wall
time was 10,138.84 seconds. Runs used the recorded NVIDIA GeForce RTX 5090,
FP32/X64-disabled JAX 0.10.2 environment and the verified GPU lock.

The fixed endpoint and structural/runtime results are reported in
[`analysis_attempt03`](../R004_formal_attempt01_analysis_attempt03/summary.md).
The earlier analysis attempts remain immutable development computations. Attempt
02 added the required printed full-history and separate runtime-outcome summaries;
attempt 03 stores the otherwise identical 35.5 MB runtime ledger as a deterministic
1.77 MB gzip payload. Neither refinement changes a training artifact or endpoint.

Deterministic full checkpoint, continuation-analysis, runtime-array, and compact
review bundles are published in the experiment-scoped
[`experiment/R004` release](https://github.com/trentonstrong/recurrent-circuit-learning-exps/releases/tag/experiment/R004).
All five assets were downloaded after publication and matched their local byte
lengths and SHA-256 identities. The complete identities and durable URLs are
recorded in [`release-assets.json`](release-assets.json).
