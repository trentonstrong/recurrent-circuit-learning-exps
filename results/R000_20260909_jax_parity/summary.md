# R000 source capture and parity

Status: passed on 2026-09-09.

The vendored notebook is byte-identical to the pinned source: 201,717 bytes,
SHA-256 `a9b3829db0d9fe0eb46148d516c18358aa648e3538aeaa682003b77cfbe757c8`.
The CPU oracle uses JAX/JAXLIB 0.4.33 and Optax 0.2.4. The RTX 5090 runtime uses
JAX/JAXLIB/CUDA plugin 0.10.2 and Optax 0.2.8. Both environments use Python
3.11.6 and have independent exact uv locks.

All wiring, Boolean gate tables, zero-boundary patch order, and hard 20-tick
executions agree exactly. FP64 gate probes agree exactly. With identical oracle
arrays, the GPU FP32 identity-biased soft trajectory first differs at tick 1
and reaches maximum absolute error `5.9008598e-6`; the nontrivial-logit trajectory
also first differs at tick 1 and reaches `1.1920929e-7`. Parameter-gradient
maximum errors are `2.861023e-6` and `5.1110983e-6`, respectively.

The identity-biased first AdamW update differs by at most `3.0517578e-5`.
This uses the declared update-specific `atol=5e-5, rtol=1e-4`: modern CPU and
GPU probes showed that Adam normalization amplifies very small gradient changes,
while the loss and gradients themselves pass the base FP32 contract. The
nontrivial update passes the base `atol=1e-6, rtol=1e-4` contract.

Modern JAX defaults `jax_threefry_partitionable=true`, changing splits,
permutations, wiring, and batches for the same seed. The implementation fixes it
to `false`, matching JAX 0.4.33; all saved and regenerated RNG arrays then agree.
Checkpoint resumption is exact for model parameters, optimizer state, model and
data RNGs, update index, and the resumed input/next update.

The isolated R001-shape GPU profile retained the notebook deterministic XLA flag.
Compile time was 12.16 s; five synchronized updates had a 101.4 ms median; peak
allocator use was 4,259,354,112 bytes. The profiling state was discarded. One
GPU process remains the initial scheduling policy.

Detailed evidence is in [`parity_report.json`](parity_report.json),
[`checkpoint_report.json`](checkpoint_report.json), and [`profile.json`](profile.json).
The CPU fixture is `tests/fixtures/r000_jax_0_4_33_cpu.npz` with SHA-256
`62379358548ad4b3052cf181ba4e74f3468d90fe7a431f6ebc9b501b1974d66d`.

Run R001 with:

```sh
uv run --project envs/jax-gpu --locked python scripts/r001.py \
  --run-id R001_$(date -u +%Y%m%dT%H%M%SZ)_seed23_jaxgpu_attempt01
```
