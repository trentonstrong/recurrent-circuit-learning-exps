# Recurrent circuit learning experiments

Can gradient search discover compact, complete descriptions of sequences with
known generators? This project studies how a circuit language and its continuous
parameterization affect learning, neutral movement, discretization, and the
structures that emerge during optimization. Increment and other small generators
are stepping stones toward harder cases such as pi.

**Status, 2026-09-12:** R000/R001 parity and the recurrent positive control,
the formal R002 paired sweep and reviews, and the fixed R003 same-function
mechanism diagnostic are complete. R003 finds representative-dependent local
SGD and visible-output response at fixed effective truth tables; it is not a
training-improvement result. R004's released continuation is independently
reviewed: exact primary-probe solutions increase from 4/64 to 16/64, while
extracted circuits generally grow and many fitted relaxations fail hardening.
R004's implementation and result records are now committed. R005's stochastic
neutral-exploration experiment and R006's continuation to 10,000 updates with
frozen hardening probes are specified for implementation; neither new
experiment has run.

## Start here

1. Read [the current research state](docs/STATE.md) and [agent instructions](AGENTS.md).
2. Review [R000: source capture and parity](experiments/R000/SPEC.md).
3. Review [R001: the synchronous recurrent baseline](experiments/R001/SPEC.md).
4. Continue through the completed [R002](experiments/R002/SPEC.md) and
   [R003](experiments/R003/SPEC.md) evidence before proposing another intervention.
5. Read the [R004 continuation review](reviews/R004_continuation/REVIEW.md)
   alongside its [frozen handoff](experiments/R004/HANDOFF.md).
6. Use the [R005 stochastic-exploration handoff](experiments/R005/HANDOFF.md)
   for the separate geometry experiment and GPU cost measurement.
7. Use the [R006 continuation and hardening handoff](experiments/R006/HANDOFF.md)
   to test a larger training budget and the approach to Boolean extraction.

The first positive control is the published DiffLogic CA synchronous checkerboard
task. It establishes recurrent learning before we adapt the architecture to
sequence generation. It is not itself a sequence-compression experiment.

The [full handoff](docs/RECURRENT_BASELINE_HANDOFF.md) explains the reference,
gate dimensionality, neutral fibers, controls, and trajectory instrumentation.
The [source manifest](references/difflogic_ca/source_manifest.json) pins the
audited notebook and extracted recipes. [Decisions](docs/DECISIONS.md) record
the reasoning behind the current plan.

## Office workstation

The measured machine has an RTX 5090 with 32,607 MiB GPU memory, a Threadripper
9970X with 32 physical / 64 logical cores, and 128 GB system RAM. It runs native
Arch Linux with kernel 7.1.8-arch1-3 and NVIDIA driver 610.57.04. See the hardware
profile and run manifests for exact provenance.

Read [workstation setup](docs/WORKSTATION_SETUP.md) before installing dependencies.
The notebook's historical JAX environment and the modern GPU runtime are distinct
profiles. R000 measured one reference process; keep that initial scheduling
policy until later experiment concurrency is profiled explicitly.

## First local Codex instruction

> Read AGENTS.md, docs/STATE.md, and experiments/R000/SPEC.md. Implement R000 and
> the R001 runner, establishing the native JAX reference first. Record the actual
> workstation environment and lock the successful dependencies. If implementing
> PyTorch, validate it against identical JAX fixtures before interpreting its
> training. After R000 passes, run the canonical R001 seed-23 experiment with the
> specified 500 updates. Commit the implementation, focused checks, run manifest,
> and small result summaries; record durable paths and checksums for large
> artifacts. Report deviations and failures. Keep R002 and later interventions
> separate from this reference run.

The repo is the shared source of truth for specifications, code, decisions, and
the [result ledger](results/README.md). Large training artifacts live outside git
and are identified by stable locations and checksums in their run manifests.

## Reproduce R000 and run R001

```sh
uv sync --project envs/jax-cpu --locked
uv sync --project envs/jax-gpu --locked

JAX_ENABLE_X64=1 uv run --project envs/jax-cpu --locked \
  pytest -q

XLA_FLAGS=' --xla_gpu_deterministic_ops=true' JAX_ENABLE_X64=1 \
  uv run --project envs/jax-gpu --locked \
  python scripts/r000.py verify-fixture \
  tests/fixtures/r000_jax_0_4_33_cpu.npz

uv run --project envs/jax-gpu --locked python scripts/r001.py \
  --run-id R001_$(date -u +%Y%m%dT%H%M%SZ)_seed23_jaxgpu_attempt01
```

The implementation fixes `jax_threefry_partitionable=false`, matching JAX
0.4.33's PRNG behavior under modern JAX. Do not remove that compatibility setting:
modern JAX otherwise produces different wiring and training batches for seed 23.
The completed result is summarized in
[the R001 result ledger](results/R001_20260909T231016Z_seed23_jaxgpu_attempt01/summary.md).

Render a saved probe trajectory as an SVG contact sheet with soft and native-hard
states at five evenly spaced runtime ticks:

```sh
uv run --project envs/jax-cpu --locked python scripts/render_checkerboard.py \
  artifacts/R001_20260909T231016Z_seed23_jaxgpu_attempt01/probe_trajectory_update_500.npz
```

Use `--ticks all`, `--mode hard`, `--sample`, `--channel`, or `--output` to
customize the rendering.
