# R001 synchronous recurrent positive control

Status: completed on 2026-09-09. This is the canonical seed-23 attempt 01; it
ran the frozen 500-update budget without retries or interventions.

The run used clean code commit `d46a6b8c838a24b6820d4181285abbff6752231c`,
configuration SHA-256
`091821c562e28262f780ba584fc5c1ee5009f27af969530db26df9c1452851d3`,
and GPU-lock SHA-256
`7a8e627cd9851d7426502bc3734bcedcfe48277575aec363877d4e7b717b2125`.
It retained the pinned deterministic XLA flag and source-compatible Threefry
behavior on JAX/JAXLIB 0.10.2, Optax 0.2.8, Python 3.11.6, and the RTX 5090.

## Outcome

The source-compatible update-500 training batch had pre-update soft loss
`0.0103987921` and native-hard loss `0`. The separate fixed 32-grid probe was
never used for training, tuning, checkpoint selection, or early stopping.

| Update | Probe soft SSE | Hard bit errors | Perfect grids |
| ---: | ---: | ---: | ---: |
| 0 | 3700.2612 | 4096 | 0/32 |
| 50 | 2064.3872 | 4096 | 0/32 |
| 100 | 2046.0139 | 4096 | 0/32 |
| 150 | 2041.1954 | 4096 | 0/32 |
| 200 | 864.4678 | 916 | 0/32 |
| 250 | 74.4066 | 0 | 32/32 |
| 300 | 2.9443 | 0 | 32/32 |
| 350 | 0.8200 | 0 | 32/32 |
| 400 | 0.1641 | 0 | 32/32 |
| 450 | 0.1795 | 0 | 32/32 |
| **500** | **0.1816** | **0** | **32/32** |

The fixed probe first became native-hard exact at saved checkpoint 250 and
remained exact through the prespecified primary checkpoint at update 500. The
small soft-loss increase after update 400 is retained; it did not trigger model
selection or a protocol change.

This establishes one working recurrent positive-control seed. It is not a
success-rate estimate, a counter/sequence-generation result, or evidence about
description length.

## Runtime and artifacts

Training compilation took 12.1781 s; evaluation compilation took 5.0990 s.
The 500 synchronized updates took 50.4346 s in aggregate with a 100.64 ms median.
The full loop including checkpoint evaluation took 54.8420 s. Peak allocator
use was 4,259,521,280 bytes.

The committed [`manifest.json`](manifest.json) contains environment, hardware,
RNG, metric, checkpoint, and SHA-256 provenance. [`metrics.jsonl`](metrics.jsonl)
contains all 500 pre-update training records. The 6.8 MB of checkpoints, selected
soft/hard trajectories, fixed evaluation arrays, and exported final hard circuit
remain under the manifest's absolute `artifacts/` paths and are intentionally
excluded from git. The final checkpoint was reloaded successfully at update 500,
and its four reported artifact hashes were verified after completion.
