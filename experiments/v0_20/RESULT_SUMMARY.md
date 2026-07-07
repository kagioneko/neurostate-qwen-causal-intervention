# NeuroState Qwen Causal Intervention v0.20 Result Summary

v0.20 scans token position while keeping the strongest v0.19 layer fixed.

## Setup

- model: Qwen2.5-0.5B-Instruct
- source layer for semantic controls: 11
- target layer: 13
- token positions: `-1,-2,-4,-8`
- train prompts: IDs `10` through `19`
- evaluation prompts: IDs `0` through `9`
- semantic controls: certainty, sentiment, verbosity, politeness
- random controls: 15 residual-space random directions
- random seed: `20260720`

## Proceed-over-Hesitate Result

| position | direction | mean slope | sign-flip p | random abs >= residual |
|---:|---|---:|---:|---:|
| -1 | neurostate | +0.125166 | 0.001953 | - |
| -1 | neurostate_residual | +0.159007 | 0.001953 | 0/15 |
| -1 | neurostate_semantic_projection | -0.126432 | 0.001953 | - |
| -2 | neurostate | +0.003554 | 0.078125 | - |
| -2 | neurostate_residual | +0.004859 | 0.025391 | 8/15 |
| -2 | neurostate_semantic_projection | -0.002605 | 0.136719 | - |
| -4 | neurostate | -0.001781 | 0.060547 | - |
| -4 | neurostate_residual | -0.000846 | 0.439453 | 13/15 |
| -4 | neurostate_semantic_projection | -0.005031 | 0.001953 | - |
| -8 | neurostate | +0.015328 | 0.142578 | - |
| -8 | neurostate_residual | +0.015591 | 0.144531 | 5/15 |
| -8 | neurostate_semantic_projection | +0.002400 | 0.796875 | - |

## Interpretation

The v0.19 layer-13 residual effect is highly position-specific. At the final
token position (`-1`), `neurostate_residual` strongly increases the
proceed-over-hesitate contrast and exceeds all 15 residual-space random
controls by absolute slope.

The same residual direction does not retain comparable strength farther back in
the prompt. At positions `-2`, `-4`, and `-8`, the effect is much smaller and is
not clearly separated from random residual controls.

This supports a boundary-local interpretation: the NeuroState residual appears
to act most strongly at the immediate response boundary in Qwen layer 13, rather
than as a broad prompt-wide direction.

## Source Artifacts

The Colab output archive was downloaded as:

```text
neurostate_qwen_causal_intervention_v0_20_results.zip
```

The extracted analysis files are under:

```text
colab_results/analysis_position_scan/
```
