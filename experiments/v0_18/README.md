# NeuroState Qwen Causal Intervention v0.18

This experiment is the reverse-split counterpart to v0.17. It tests whether the
NeuroState residual effect replicates when the semantic controls are learned on
the second prompt split and evaluated on the first prompt split.

Default reverse split:

- train prompts: IDs `10` through `19`
- evaluation prompts: IDs `0` through `9`

The semantic controls are certainty, sentiment, verbosity, and politeness. Each
control is learned from paired instructions at Qwen layer 11 using only the train
split. NeuroState is then decomposed into:

- `neurostate_semantic_projection`: the component explained by the four learned
  controls.
- `neurostate_residual`: the component orthogonal to those controls.

The scan evaluates NeuroState, the controls, the projection, the residual, and
15 fresh residual-space random controls on the held-out evaluation prompts. All
directions are injected into layer 13 at the official Qwen chat response
boundary, position `-1`.

Together with v0.17, this forms a two-way cross-validation of the residual
response-boundary effect.

Upload `neurostate_qwen_causal_intervention_v0_18_reverse_split_colab.ipynb`,
select a T4 GPU runtime, and run all cells.
