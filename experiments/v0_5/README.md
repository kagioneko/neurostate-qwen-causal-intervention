# NeuroState Qwen Causal Intervention v0.5

This experiment builds a Qwen2.5-0.5B direction directly from paired
proceed-vs-hesitate prompts, then tests whether the sharper direction improves
specificity at the strongest v0.4 site.

## Primary test

- Build `qwen_proceed_hesitate_direction_v0_5.json` from paired prompt
  activations at decoder layer 13.
- Base intervention prompts contain no NeuroState memo.
- Compare decoder layers 7 and 13.
- Inject the unit direction at each layer.
- Sweep alpha over `-2, -1, -0.5, 0, 0.5, 1, 2`.
- Measure the primary proceed-vs-hesitate next-token contrast and generated wording.
- Keep the older action-vs-calm contrast in the JSONL as a secondary comparison.
- Compare the semantic-direction slope with norm-matched random directions
  orthogonal to the semantic direction.
- Export the full random-slope null distribution so weak specificity can be
  diagnosed instead of reduced to a single rank-p value.

The layer 7 vs 13 check tests the v0.4 observation that layer 7 can show an
opposite-sign effect while layer 13 is the strongest positive intervention
site. The optional full run should use layer 13 if the layer comparison
preserves that pattern.

This tests causal influence on model policy. It does not establish emotion,
conscious state, or universal cross-model geometry.

## What v0.5 adds

v0.4 confirmed that layer 13 is steerable, but specificity was moderate because
9 of 63 random orthogonal controls beat the semantic direction by absolute
slope. v0.5 tests whether a direction built directly from proceed/hesitate
pairs is sharper than the inherited cross-architecture direction:

- `build_direction.py` creates a new paired-prompt direction at layer 13.
- The first intervention run compares layers 7 and 13 with 15 random controls.
- The optional full run confirms layer 13 with 63 random controls.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_5_colab.ipynb`, select a T4 GPU,
and run all cells. Start with direction build and the layer 7 vs 13 comparison.
Run the optional full cell only if layer 13 remains the best positive site.
