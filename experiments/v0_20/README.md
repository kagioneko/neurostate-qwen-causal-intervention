# NeuroState Qwen Causal Intervention v0.20

This experiment scans where the NeuroState residual effect is located around the
official Qwen chat response boundary. It keeps the strongest v0.19 layer
(`layer 13`) fixed, and injects each direction into multiple token positions.

Default split:

- train prompts: IDs `10` through `19`
- evaluation prompts: IDs `0` through `9`

The semantic controls are certainty, sentiment, verbosity, and politeness. Each
control is learned from paired instructions at Qwen layer 11 using only the train
split. NeuroState is then decomposed into:

- `neurostate_semantic_projection`: the component explained by the four learned
  controls.
- `neurostate_residual`: the component orthogonal to those controls.

The scan evaluates NeuroState, the controls, the projection, the residual, and
15 fresh residual-space random controls on the held-out evaluation prompts.

Position scan:

- target layer: `13`
- token positions: `-1,-2,-4,-8`
- prompt format: official Qwen chat template with generation prompt

The goal is to determine whether the residual proceed-over-hesitate effect is
response-boundary-specific or remains stable farther back in the prompt.

## Related Context

Anthropic announced `J-space` and `J-lens` on 2026-07-06, describing a hidden
internal workspace-like structure whose latent concepts can causally affect
model outputs. This experiment should not be treated as the same method or the
same phenomenon: it uses Qwen hidden-state direction injection, not Anthropic's
Jacobian lens.

However, the connection is conceptually relevant. The v0.16-v0.19 results suggest
that NeuroState is not just an ordinary surface semantic direction such as
certainty, sentiment, verbosity, or politeness. Instead, it may be acting on a
boundary-sensitive internal control component. That makes J-space/J-lens an
important related interpretability context for later discussion.

Upload `neurostate_qwen_causal_intervention_v0_20_residual_position_scan_colab.ipynb`,
select a T4 GPU runtime, and run all cells.
