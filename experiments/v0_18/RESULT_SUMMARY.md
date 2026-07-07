# NeuroState Qwen Causal Intervention v0.18 Result Summary

Colab result ZIP:

- `C:\Users\sakih\Downloads\neurostate_qwen_causal_intervention_v0_18_results.zip`

Extracted result directory:

- `D:\NeuroState_Qwen_Causal_Intervention_v0_18\colab_results`

## Design

v0.18 is the reverse-split counterpart to v0.17.

- train prompt IDs: `10` through `19`
- evaluation prompt IDs: `0` through `9`
- semantic controls: certainty, sentiment, verbosity, politeness
- target: Qwen official chat template, layer `13`, position `-1`
- residual-space random controls: `15`
- random seed: `20260718`

## Key Finding

The NeuroState residual effect replicates again under the reverse split.

Geometry:

- `neurostate_residual` cosine with NeuroState: `0.971909`
- `neurostate_semantic_projection` cosine with NeuroState: `0.235359`

Primary behavior:

- `neurostate_residual` on `proceed_hesitate`: mean slope `+0.159007`, p `0.001953`
- `neurostate` on `proceed_hesitate`: mean slope `+0.125166`, p `0.001953`
- `neurostate_semantic_projection` on `proceed_hesitate`: mean slope `-0.126432`, p `0.001953`

As in v0.17, the semantic projection moves in the opposite direction on the
central proceed-over-hesitate contrast.

## Random Residual Controls

For `proceed_hesitate`, none of the 15 residual-space random controls had an
absolute mean slope as large as the NeuroState residual.

- NeuroState residual: `+0.159007`
- random residual min: `-0.132069`
- random residual max: `+0.072536`
- random controls with `abs(slope) >= abs(neurostate_residual)`: `0 / 15`

## Secondary Measures

NeuroState residual:

- `certainty_uncertainty`: `+0.131623`, p `0.001953`
- `polite_blunt`: `-0.105826`, p `0.003906`
- `positive_negative`: `+0.015780`, p `0.394531`
- `verbose_concise`: `+0.025320`, p `0.207031`

The stable pattern is proceed-over-hesitate plus certainty increase and
bluntness shift. Positive/negative and verbosity are not reliable residual
effects in the reverse split.

## Cross-Validation Interpretation

v0.17 and v0.18 together form a two-way split validation:

- v0.17: train `0-9`, evaluate `10-19`
- v0.18: train `10-19`, evaluate `0-9`

Both runs show:

- NeuroState remains mostly in the residual after removing four learned semantic
  controls.
- The residual preserves a strong positive `proceed_hesitate` effect.
- The semantic projection reverses the `proceed_hesitate` sign.
- No sampled residual-space random control exceeds the NeuroState residual on
  the primary contrast.

The strongest supported claim is now:

> NeuroState contains a boundary-sensitive activation component that survives
> removal of several ordinary semantic directions and cross-validates across
> disjoint prompt splits.

## Natural Next Experiment

v0.19 should reduce dependence on a single model-internal source vector by
testing whether the same residual pattern holds under a second learned or
derived NeuroState vector.

Possible options:

- derive a fresh NeuroState-like direction from a different prompt subset
- compare layer `12`, `13`, and `14` for the residual direction
- keep the official chat boundary and both split directions

