# NeuroState Qwen Causal Intervention v0.14

Corrected independent random-control replication of v0.12.

- v0.12 seeds: `20260712` through `20260774`
- v0.14 seeds: `20261712` through `20261774`
- overlap: zero directions

All other conditions remain fixed: official Qwen chat template, layer 13,
position -1, 20 prompts, 63 random controls, and the `proceed_hesitate`
contrast.

Upload `neurostate_qwen_causal_intervention_v0_14_disjoint_seed_colab.ipynb`,
select a T4 GPU runtime, and run all cells.
