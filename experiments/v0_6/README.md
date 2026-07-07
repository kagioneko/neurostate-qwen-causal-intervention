# NeuroState Qwen Causal Intervention v0.6

This experiment keeps the stronger inherited NeuroState direction from v0.4 and
tests where inside the prompt sequence the layer-13 intervention matters.

## Primary test

- Use `qwen_middle_prompt_direction.json`, the current best direction.
- Intervene at decoder layer `13`.
- Compare token positions `-1, -2, -4, -8`, counted from the end of the prompt.
- Sweep alpha over `-2, -1, -0.5, 0, 0.5, 1, 2` for the semantic direction.
- Compare against random orthogonal controls at each token position.
- Analyze the proceed-vs-hesitate next-token contrast.

Position `-1` is the previous last-token intervention site. Earlier positions
test whether the effect is tied to the final residual state or can be driven
from upstream prompt tokens.

This tests a local causal path from token position to output policy. It does not
establish emotion, conscious state, or universal cross-model geometry.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_6_colab.ipynb`, select a T4 GPU,
and run all cells. The main output is:

- `analysis_position_scan/position_report.md`
- `analysis_position_scan/position_summary.csv`
- `analysis_position_scan/position_summary.json`

The optional full cell is only for reproducing the v0.4 last-token full baseline
inside the same archive.
