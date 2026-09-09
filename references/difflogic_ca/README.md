# Audited DiffLogic CA reference

`source_manifest.json` is copied unchanged from the 2026-09-09 source audit.
It contains the notebook pin, full-file and selected-cell checksums, extracted
synchronous/asynchronous recipes, and derived parameter counts. It also preserves
the handoff's proposed experiment outline; current experiment status lives in
`docs/STATE.md` and the corresponding `experiments/Rxxx/SPEC.md`.

The notebook source itself is not vendored yet. R000 must retrieve and verify it
before extracting executable code, preserve its Apache-2.0 attribution, and
retain the original checksum alongside any local adaptations. This audit is not
a full environment lock and does not claim to identify the exact source revision
used for the paper's reported results.

Primary references:

- [Paper, v1](https://arxiv.org/html/2506.04912v1)
- [Official project](https://google-research.github.io/self-organising-systems/difflogic-ca/)
- [Pinned notebook](https://github.com/google-research/self-organising-systems/blob/3d5547ca48b60ecac459834e2c05c9ff5df87991/notebooks/diffLogic_CA.ipynb)
- [Light Differentiable Logic Gate Networks, v1](https://arxiv.org/html/2510.03250v1)
