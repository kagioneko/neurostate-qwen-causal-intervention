# NeuroState Qwen Causal Intervention v0.17 Result Summary

Colab result ZIP:

- `C:\Users\sakih\Downloads\neurostate_qwen_causal_intervention_v0_17_results.zip`

Extracted result directory:

- `D:\NeuroState_Qwen_Causal_Intervention_v0_17\colab_results`

## Design

v0.17 tests whether the v0.16 residual result survives a disjoint prompt split.

- train prompt IDs: `0` through `9`
- evaluation prompt IDs: `10` through `19`
- semantic controls: certainty, sentiment, verbosity, politeness
- target: Qwen official chat template, layer `13`, position `-1`
- residual-space random controls: `15`

## Key Finding

The NeuroState residual effect replicates on held-out prompts.

The residual direction remains almost identical to the inherited NeuroState
direction geometrically:

- `neurostate_residual` cosine with NeuroState: `0.973931`
- `neurostate_semantic_projection` cosine with NeuroState: `0.226845`

The residual also preserves the main behavioral effect:

- `neurostate_residual` on `proceed_hesitate`: mean slope `+0.138071`, p `0.001953`
- `neurostate` on `proceed_hesitate`: mean slope `+0.100162`, p `0.001953`
- `neurostate_semantic_projection` on `proceed_hesitate`: mean slope `-0.151685`, p `0.001953`

The semantic projection again moves in the opposite direction on the central
proceed-over-hesitate contrast.

## Random Residual Controls

For `proceed_hesitate`, none of the 15 residual-space random controls had an
absolute mean slope as large as the NeuroState residual.

- NeuroState residual: `+0.138071`
- random residual min: `-0.135626`
- random residual max: `+0.073796`
- random controls with `abs(slope) >= abs(neurostate_residual)`: `0 / 15`

This makes the held-out proceed effect stronger than the sampled residual-space
random baseline.

## Secondary Measures

NeuroState residual:

- `certainty_uncertainty`: `+0.119925`, p `0.001953`
- `polite_blunt`: `-0.107859`, p `0.001953`
- `positive_negative`: `+0.025614`, p `0.027344`
- `verbose_concise`: `-0.002199`, p `0.910156`

The positive/negative and verbosity results are not central. The stable pattern
is proceed-over-hesitate plus certainty increase and bluntness shift.

## Interpretation

v0.17 strengthens the v0.16 conclusion. The NeuroState response-boundary effect
is not explained away by learned certainty, sentiment, verbosity, or politeness
controls, and its residual component generalizes to prompts that were not used
to learn the semantic control basis.

The strongest claim supported by this run is:

> NeuroState contains a boundary-sensitive activation component that survives
> removal of several ordinary semantic directions and replicates on held-out
> prompts.

## Natural Next Experiment

v0.18 should test split reversal:

- train controls on prompt IDs `10` through `19`
- evaluate on prompt IDs `0` through `9`
- keep the same layer, position, semantic controls, and random residual controls

If the same residual effect appears in the reverse split, the result becomes a
clean two-way cross-validation.

