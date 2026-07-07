# NeuroState Qwen Causal Intervention v0.1

This experiment intervenes on the Qwen2.5-0.5B middle-layer direction that
survived held-out paraphrase transfer and exact maxT correction in the
cross-architecture probe v0.2.

## Primary test

- Base prompts contain no NeuroState memo.
- Inject the unit active-minus-calm direction at decoder layer 11.
- Sweep alpha over `-2, -1, -0.5, 0, 0.5, 1, 2`.
- Measure active-minus-calm next-token logit contrast and generated wording.
- Compare the semantic-direction slope with 63 norm-matched random directions
  orthogonal to the semantic direction.

This tests causal influence on model policy. It does not establish emotion,
conscious state, or universal cross-model geometry.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_1_colab.ipynb`, select a T4 GPU,
and run all cells. Start with smoke. The optional full cell runs the
confirmatory experiment and the final cell exports a ZIP.
