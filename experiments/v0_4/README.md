# NeuroState Qwen Causal Intervention v0.4

This experiment intervenes on the Qwen2.5-0.5B middle-layer direction that
survived held-out paraphrase transfer and exact maxT correction in the
cross-architecture probe v0.2.

## Primary test

- Base prompts contain no NeuroState memo.
- Scan decoder layers around the middle-layer target.
- Inject the unit direction at each layer.
- Sweep alpha over `-2, -1, -0.5, 0, 0.5, 1, 2`.
- Measure the primary proceed-vs-hesitate next-token contrast and generated wording.
- Keep the older action-vs-calm contrast in the JSONL as a secondary comparison.
- Compare the semantic-direction slope with norm-matched random directions
  orthogonal to the semantic direction.
- Export the full random-slope null distribution so weak specificity can be
  diagnosed instead of reduced to a single rank-p value.

The scan is for localization. The confirmatory run still uses the strongest
layer from the scan and a larger random-direction control set.

This tests causal influence on model policy. It does not establish emotion,
conscious state, or universal cross-model geometry.

## What v0.4 adds

v0.3 found the strongest full-run result so far at layer 13 with
`contrast_bank=proceed_hesitate`, but random-direction specificity was still
moderate. v0.4 keeps the same intervention protocol and adds diagnostic outputs:

- `random_slope_distribution.csv` ranks every random direction by absolute
  logit slope.
- `causal_summary.json` now reports how many random directions beat the
  semantic direction, the semantic slope quantile, and the max random slope.
- `causal_report.md` includes the top random controls for quick inspection.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_4_colab.ipynb`, select a T4 GPU,
and run all cells. Start with the layer scan. The optional full cell runs the
confirmatory experiment for the chosen layer and the final cell exports a ZIP.
