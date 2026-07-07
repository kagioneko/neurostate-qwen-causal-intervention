# NeuroState Qwen Causal Intervention v0.17

This experiment tests whether the v0.16 NeuroState residual effect replicates
when semantic controls are learned on one prompt split and evaluated on a
disjoint prompt split.

Default split:

- train prompts: IDs `0` through `9`
- evaluation prompts: IDs `10` through `19`

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

Upload `neurostate_qwen_causal_intervention_v0_17_residual_replication_colab.ipynb`,
select a T4 GPU runtime, and run all cells.
