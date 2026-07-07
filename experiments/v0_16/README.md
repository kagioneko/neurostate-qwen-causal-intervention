# NeuroState Qwen Causal Intervention v0.16

This experiment continues v0.15 by separating the inherited NeuroState direction
into two parts:

- `neurostate_semantic_projection`: the component explained by four learned
  semantic controls.
- `neurostate_residual`: the component orthogonal to those controls.

The semantic controls are certainty, sentiment, verbosity, and politeness. Each
control is learned from paired instructions at Qwen layer 11. NeuroState, the
four controls, the semantic projection, and the residual are then injected into
layer 13 at the official Qwen chat response boundary.

This tests whether the boundary effect is mostly reducible to ordinary semantic
axes, or whether an independent NeuroState component remains after those axes are
removed.

Upload `neurostate_qwen_causal_intervention_v0_16_semantic_residual_colab.ipynb`,
select a T4 GPU runtime, and run all cells.
