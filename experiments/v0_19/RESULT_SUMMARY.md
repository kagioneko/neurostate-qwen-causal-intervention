# NeuroState Qwen Causal Intervention v0.19 Result Summary

Colab result ZIP:

- `C:\Users\sakih\Downloads\neurostate_qwen_causal_intervention_v0_19_results.zip`

Extracted result directory:

- `D:\NeuroState_Qwen_Causal_Intervention_v0_19\colab_results`

## Design

v0.19 scans the target decoder layer while keeping the reverse split from v0.18.

- train prompt IDs: `10` through `19`
- evaluation prompt IDs: `0` through `9`
- semantic controls: certainty, sentiment, verbosity, politeness
- target layers: `11,12,13,14,15`
- token position: `-1`
- prompt format: official Qwen chat template
- residual-space random controls: `15`
- random seed: `20260719`

## Key Finding

The NeuroState residual `proceed_hesitate` effect is sharply layer-dependent and
peaks at layer 13.

NeuroState residual on `proceed_hesitate` by layer:

| layer | mean slope | sign-flip p | random controls with abs >= residual |
|---:|---:|---:|---:|
| 11 | `-0.008170` | `0.402344` | `14 / 15` |
| 12 | `+0.043977` | `0.005859` | `5 / 15` |
| 13 | `+0.159007` | `0.001953` | `0 / 15` |
| 14 | `-0.055713` | `0.015625` | `4 / 15` |
| 15 | `+0.002205` | `0.851562` | `15 / 15` |

Layer 13 is the only scanned layer where the residual proceed effect is both
large and stronger than all sampled residual-space random controls.

## Comparison with Original NeuroState and Projection

Original NeuroState on `proceed_hesitate`:

| layer | mean slope | sign-flip p |
|---:|---:|---:|
| 11 | `-0.031526` | `0.035156` |
| 12 | `+0.012565` | `0.457031` |
| 13 | `+0.125166` | `0.001953` |
| 14 | `-0.091195` | `0.001953` |
| 15 | `-0.016408` | `0.169922` |

Semantic projection on `proceed_hesitate`:

| layer | mean slope | sign-flip p |
|---:|---:|---:|
| 11 | `-0.103136` | `0.001953` |
| 12 | `-0.126341` | `0.001953` |
| 13 | `-0.126432` | `0.001953` |
| 14 | `-0.158165` | `0.001953` |
| 15 | `-0.080603` | `0.001953` |

The semantic projection consistently pushes against proceed-over-hesitate across
all scanned layers. The residual does not simply mirror that projection; it
produces a specific positive peak at layer 13.

## Interpretation

v0.19 strengthens the view that the NeuroState residual is not just a generic
direction that works anywhere in the model. The main proceed-over-hesitate
effect appears localized around the previously selected intervention layer,
layer 13.

The strongest supported claim after v0.19 is:

> NeuroState contains a boundary-sensitive activation component that survives
> removal of ordinary semantic controls, cross-validates across prompt splits,
> and is most effective when injected at Qwen decoder layer 13.

This is consistent with a layer-specific internal control point rather than a
diffuse surface semantic shift.

## Related Context

The separate `RELATED_J_SPACE_NOTE.md` records the conceptual connection to
Anthropic's 2026-07-06 J-space/J-lens announcement. This experiment is not a
J-lens replication, but it is related in the broad sense that both lines of work
probe latent internal structures that can causally affect model behavior.

## Natural Next Experiment

v0.20 should test whether the same layer-13 residual effect holds under token
position perturbation around the official chat boundary:

- target layer fixed to `13`
- positions: `-1,-2,-4,-8`
- same reverse split
- same residual-space random controls

If the effect is strongest at position `-1`, that would jointly localize it to
both a layer and a response-boundary token.

