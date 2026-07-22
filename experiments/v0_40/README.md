# NeuroState v0.40

This snapshot consolidates the multi-model three-axis audit, the Qwen3 approach/refusal holdout audit, the negative vigilance follow-up, and an interactive approach-steering demo.

## Main findings

- Six descriptive dimensions remain useful for prompt construction, but were not established as six independent causal internal controls.
- Approach was the clearest reproducible intervention direction. On the unused Qwen3-1.7B holdout it moved in the expected direction on 20/20 prompts.
- At source layer 18, approach and the 12-pair refusal proxy had cosine -0.0247. This is a layer-specific result, not global independence: the 28-layer scan ranged from -0.3493 at layer 7 to -0.0186 at layer 19.
- Vigilance did not separate from random controls on either the earlier action-stopping measure or the false-premise single-token follow-up (p = 0.455).
- The steering demo compares the semantic approach direction with five orthogonal random directions. It fixes the evaluated intervention at source 18 -> target 20.

## Reproduction files

- `Qwen3_Approach_Refusal_Holdout_Audit_Colab.ipynb`
- `Qwen3_Vigilance_False_Premise_Audit_Colab.ipynb`
- `Qwen3_Approach_Steering_Demo_Colab.ipynb`

The notebooks upload their matching `colab_*.py` files. The steering demo also requires `colab_neurostate_3axis.py`.

## Reading order

1. `NEUROSTATE_3AXIS_RESULTS_JA.md`
2. `FINAL_DECISION_JA.md`
3. `PRIMARY_SOURCES.md`
4. `../../articles/NOTION_ARTICLE_NEUROSTATE_BEGINNER_JA.md`
5. `../../articles/APPROACH_STEERING_DEMO_GUIDE_JA.md`

No model weights, private logs, memory text, VPS data, or raw result ZIP archives are included.
