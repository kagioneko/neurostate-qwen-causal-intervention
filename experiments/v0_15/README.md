# NeuroState Qwen Causal Intervention v0.15

This experiment compares the inherited NeuroState direction with four learned
semantic controls: certainty, sentiment, verbosity, and politeness.

Each control is learned from paired instructions at Qwen layer 11 and then
injected, like NeuroState, into layer 13 at the official chat response boundary.
All five directions use 20 evaluation prompts and seven alpha values.

The analysis reports slopes across five output contrasts and cosine similarity
between every control and NeuroState. This tests what kind of semantic behavior
the robust boundary steering most closely resembles.

Upload `neurostate_qwen_causal_intervention_v0_15_semantic_controls_colab.ipynb`,
select a T4 GPU runtime, and run all cells.
