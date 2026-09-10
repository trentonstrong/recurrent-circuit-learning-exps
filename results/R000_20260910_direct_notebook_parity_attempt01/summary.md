# R000 direct notebook execution parity

Status: passed on 2026-09-10. This is implementation/runtime evidence, not a
training result.

The oracle executed cells 6, 8, 10, 12, and 16 from the vendored notebook in an
isolated namespace under historical JAX/JAXLIB 0.4.33 on CPU. The report retains
the complete executed source, cell IDs and hashes, supplied globals, adapters,
and notebook SHA-256
`a9b3829db0d9fe0eb46148d516c18358aa648e3538aeaa682003b77cfbe757c8`.
X64 was disabled and `jax_threefry_partitionable` was false.

The historical-CPU extraction comparison passed. Wiring, initialized parameters,
model and data keys, three successive sampled batches, Boolean execution, hard
gate IDs, hard trajectories, soft trajectories, losses, optimizer state, and
updated parameters matched exactly. The only nonzero comparison was an
independently recomputed identity step-gradient path at `7.4505806e-9`, within
the FP32 policy.

The same actual notebook arrays passed against modern JAX/JAXLIB 0.10.2 on the
RTX 5090. Seeded arrays and hard execution were exact. Maximum observed absolute
errors included `5.8412552e-6` for the identity soft trajectory,
`5.1110983e-6` for nontrivial gradients, `3.0517578e-5` for the sensitive
identity AdamW update, and `3.3569336e-4` for the summed nontrivial loss. Every
comparison passed the pre-existing R000 tolerance policy; no sampler or tolerance
was changed.

The first verifier invocation is retained as `failed_verifier_attempt.json`. It
failed before numerical comparison because the schema-v1 verifier assumed every
fixture included FP64 arrays. The X64-disabled schema-v2 fixture intentionally
keeps those diagnostics separate. Making those fields optional resolved the
harness issue without regenerating the oracle fixture.

The 1,634,432-byte oracle fixture remains outside git at
`artifacts/R000_20260910_direct_notebook_parity_attempt01/notebook_oracle_x64_disabled.npz`
with SHA-256 `1ed579efdc49dcb60fdccf9543960f55bc6557a252e68018c900d80e284aa898`.
See `notebook_oracle_report.json`, `cpu_extraction_parity_report.json`, and
`gpu_runtime_parity_report.json` for complete comparisons.
