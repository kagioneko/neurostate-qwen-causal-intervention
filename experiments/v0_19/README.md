# NeuroState Qwen Causal Intervention v0.19

This experiment scans which decoder layers carry the NeuroState residual effect.
It uses the reverse split from v0.18 and injects each direction into multiple
layers at the official Qwen chat response boundary.

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

Layer scan:

- target layers: `11,12,13,14,15`
- token position: `-1`
- prompt format: official Qwen chat template with generation prompt

The goal is to determine whether the residual proceed-over-hesitate effect is
layer-13-specific or spreads across neighboring layers.

## Related Context

Anthropic announced `J-space` and `J-lens` on 2026-07-06, describing a hidden
internal workspace-like structure whose latent concepts can causally affect
model outputs. This experiment should not be treated as the same method or the
same phenomenon: it uses Qwen hidden-state direction injection, not Anthropic's
Jacobian lens.

However, the connection is conceptually relevant. The v0.16-v0.18 results suggest
that NeuroState is not just an ordinary surface semantic direction such as
certainty, sentiment, verbosity, or politeness. Instead, it may be acting on a
boundary-sensitive internal control component. That makes J-space/J-lens an
important related interpretability context for later discussion.

Upload `neurostate_qwen_causal_intervention_v0_19_residual_layer_scan_colab.ipynb`,
select a T4 GPU runtime, and run all cells.
