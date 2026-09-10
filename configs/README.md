# Configuration contract

These JSON files are validated experiment specifications and runner inputs.

- `experiments/r001_sync_reference.json` separates the audited scientific recipe
  from additional project evaluation and runtime selection.
- `experiments/r002_paired_gate_coordinates.json` freezes the four-condition,
  16-seed paired comparison and its shared evaluation/runtime contract.
- `hardware/office_5090.json` records owner-reported hardware and unresolved local
  environment fields. It is not a dependency lock or a hardware benchmark.

Preserve the original `reference_recipe` fields. Actual installed versions belong
in a resolved environment profile and each run manifest. A runner must validate
its configuration and record the hash of the exact serialized config used.

The full source audit lives in `references/difflogic_ca/source_manifest.json` at
the repository root. It retains the historical package pins even when runtime
adaptation is necessary.
