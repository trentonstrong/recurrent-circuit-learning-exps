# Office workstation setup

Status: environment strategy implemented and R000 GPU profile completed on
2026-09-09. See the R000 result for measured parity and performance.

## Hardware and open items

| Item | Recorded value | Evidence |
| --- | --- | --- |
| GPU | NVIDIA RTX 5090, 32,607 MiB reported by driver | `nvidia-smi` during R000 |
| GPU architecture | Blackwell, compute capability 12.0 (`sm_120`) | [NVIDIA table](https://developer.nvidia.com/cuda/gpus) |
| CPU | AMD Threadripper 9970X, 32 physical / 64 logical cores | `lscpu` during R000 |
| System memory | 134,379,921,408 bytes; ECC owner-reported | `free -b` and owner report |
| OS | Native Arch Linux, x86_64 | Local inspection |
| Python environment manager | uv 0.12.5 | Local inspection |
| Kernel; NVIDIA driver | 7.1.8-arch1-3; 610.57.04 | Local inspection and `nvidia-smi` |

The [machine profile](../configs/hardware/office_5090.json) records the R000
measurements. Continue to capture device, driver, OS, Python, package, and backend
versions in each run manifest because host state can change.

## Three environment roles

| Role | Purpose | Version policy |
| --- | --- | --- |
| Historical JAX CPU | Small reference fixtures and numerical oracle | JAX/JAXLIB 0.4.33; resolve and lock compatible Python and remaining dependencies |
| Modern JAX GPU | Native implementation and canonical training on the 5090 | Stable Blackwell-compatible release; lock after R000 checks |
| Optional PyTorch | Port and later research | Separate environment; compare identical fixtures to JAX before training interpretation |

Create these as independent uv projects during R000: `envs/jax-cpu` and
`envs/jax-gpu`, plus `envs/torch-gpu` if implementing the port. Their distinct JAX
pins require separate dependency resolutions. uv documents
[independent projects with path dependencies](https://docs.astral.sh/uv/concepts/projects/workspaces/)
for conflicting requirements; shared repository code can be a local path dependency.
These environment projects and locks have not been created yet.

A seed alone cannot establish
parity between them. Preserve and compare the actual wiring, initial parameters,
inputs, targets, RNG semantics, losses, and updates.

The old notebook environment is historical evidence, not a verified RTX 5090
installation recipe. Use the source pin to hold the algorithm fixed and report
the runtime adaptation. Do not replace the audited version fields with the new
installed versions; put installed versions in an environment lock and run manifest.

## Current package guidance

[JAX installation guidance](https://docs.jax.dev/en/latest/installation.html)
recommends CUDA 13 pip wheels (`jax[cuda13]`) and a Linux driver at least version
580. Use the Linux x86_64 wheels on this native Arch workstation. Select exact
package versions during R000 and commit the validated lock; this handoff is not
a resolved install script.

For a PyTorch port, the [PyTorch 2.12 release guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/)
recommends CUDA 13.0+ wheels for Blackwell. That guidance specifies drivers at least
580.65.06 on Linux. Recheck the selected release's requirements
when resolving the environment. Do not infer GPU support merely from importing
the package or from the CUDA version displayed by `nvidia-smi`.

Verify that the selected backend sees the 5090, executes the relevant forward
and gradient computations on it, and passes reference parity. Capture failures
and necessary API adaptations. Retain the reference deterministic XLA flag where
supported; if a modern runtime rejects it, record the issue and validated
replacement or limitation instead of silently discarding it.

## Arch environment choices

Use [uv](https://docs.astral.sh/uv/guides/install-python/) to manage and pin the
Python interpreter for each experiment environment. Resolve a Python version
compatible with the historical JAX wheels and a version supported by the selected
modern GPU stack; they need not be the same. Each environment project must commit
its `pyproject.toml`, `uv.lock`, and `.python-version` with an exact Python patch
version after validation. Keep its `.venv` out of git.

For reproduction, use `uv sync --locked` and `uv run --locked` from the selected
environment project. The [locked mode](https://docs.astral.sh/uv/concepts/projects/sync/)
checks that the existing lock matches the project instead of updating it during
a run. Dependency upgrades belong to a recorded environment revision. Capture
the uv version, interpreter version, and lock checksum in each run manifest.

Start local inspection with these read-only commands:

```sh
uname -r
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv --version
```

Record the installed kernel and NVIDIA package versions too. Inspect the existing
working driver before deciding whether any system change is necessary. The JAX
CUDA pip-wheel route supplies user-space CUDA dependencies; select its matching
driver requirement during R000. A local CUDA toolkit is a separate need if later
work compiles extensions that require it.

After a kernel, driver, interpreter, or dependency change, record a new environment
identity and rerun focused device/parity checks before pooling new runs with old
ones. Package locks do not capture the host kernel and NVIDIA driver, so retain
that metadata in each run manifest.

## Resource policy for the first runs

Start with one GPU training process and the unchanged FP32 reference recipe.
This machine is a plausible platform for the reference-scale experiments, but
actual capacity and throughput depend on compiled operations and saved backward
activations, not only on parameter count.

JAX normally [preallocates 75% of GPU memory](https://docs.jax.dev/en/latest/gpu_memory_allocation.html).
Several simultaneous default processes can exhaust memory before useful work.
Record allocator settings, profile one run, and leave room for display or other
workstation use. Choose memory fractions or disable preallocation only as an
explicit, recorded runtime setting; allocation behavior affects fragmentation.

Record compile time separately from steady-state update time. Synchronize device
work before timing it; asynchronous dispatch otherwise gives misleading numbers.
Use backend memory reports alongside process-level GPU memory, since reserved
allocator memory and live tensors are different quantities. Exclude profiling
warm-up from the canonical trajectory by restoring model, optimizer, and RNG
state before update 1.

Use the CPU for reference fixtures, analysis, and artifact handling. Synthetic
checkerboard batches do not require a large data-loader worker pool. Select
thread counts by measurement rather than assuming 64 workers improve throughput.

FP64 is useful for small derivative diagnostics. Mixed precision, custom fused
kernels, changes to gate algebra, and alternative compilation settings require
named, validated follow-up interventions. Do not change batch size, unroll, grid,
or precision merely to fit more seeds into memory in the baseline comparison.

## What R000 must leave behind

- Separate reproducible environment specifications for the CPU oracle and GPU
  runner, including Python, JAX/JAXLIB, Optax, Flax where used, and CUDA packages.
- Actual OS, device, driver, precision, compiler, and allocator metadata.
- Verified source checksum, saved fixtures, focused parity report, and any source
  API adaptations with their effects assessed.
- One-run time/memory profile and a justified initial scheduling choice.

There is no measured runtime estimate or validated dependency lock yet.
