# Recurrent circuit learning experiments

Can gradient search discover compact, complete descriptions of sequences with
known generators? This project studies how a circuit language and its continuous
parameterization affect learning, neutral movement, discretization, and the
structures that emerge during optimization. Increment and other small generators
are stepping stones toward harder cases such as pi.

**Status, 2026-09-09:** research handoff and experiment specifications only.
The official DiffLogic CA source has been audited. This repository does not yet
contain a learner, a working environment lock, or new training results.

## Start here

1. Read [the current research state](docs/STATE.md) and [agent instructions](AGENTS.md).
2. Implement [R000: source capture and parity](experiments/R000/SPEC.md).
3. Run [R001: the synchronous recurrent baseline](experiments/R001/SPEC.md).
4. Review that evidence before launching [R002](experiments/R002/SPEC.md) or
   [R003](experiments/R003/SPEC.md).

The first positive control is the published DiffLogic CA synchronous checkerboard
task. It establishes recurrent learning before we adapt the architecture to
sequence generation. It is not itself a sequence-compression experiment.

The [full handoff](docs/RECURRENT_BASELINE_HANDOFF.md) explains the reference,
gate dimensionality, neutral fibers, controls, and trajectory instrumentation.
The [source manifest](references/difflogic_ca/source_manifest.json) pins the
audited notebook and extracted recipes. [Decisions](docs/DECISIONS.md) record
the reasoning behind the current plan.

## Office workstation

The intended machine has an RTX 5090 with 32 GB GPU memory, a Threadripper
9970X with 32 physical / 64 logical cores, and 128 GB ECC system RAM.
These are owner-reported specifications, not measurements from this repository.
The workstation runs native Arch Linux. Driver and kernel versions remain to be
recorded locally.

Read [workstation setup](docs/WORKSTATION_SETUP.md) before installing dependencies.
The notebook's historical JAX environment and the modern GPU runtime are distinct
profiles. Begin with one GPU training process; measure activation memory and
steady-state runtime before choosing seed concurrency.

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
