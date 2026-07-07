# NeuroState Qwen Causal Intervention v0.16 Result Summary

Colab result ZIP:

- `C:\Users\sakih\Downloads\neurostate_qwen_causal_intervention_v0_16_results.zip`

Extracted result directory:

- `D:\NeuroState_Qwen_Causal_Intervention_v0_16\colab_results`

## Key Finding

The inherited NeuroState direction is not mostly explained by the four learned
semantic controls used in v0.16.

After projecting NeuroState onto the control basis of certainty, sentiment,
verbosity, and politeness, the orthogonal residual still has cosine `0.973931`
with the original NeuroState direction. The semantic projection has cosine only
`0.226845` with NeuroState.

This means the four tested semantic axes explain only a small component of the
NeuroState vector geometry.

## Direction Geometry

Cosine with NeuroState:

- `neurostate`: `1.000000`
- `certainty`: `0.207848`
- `sentiment`: `0.109058`
- `verbosity`: `-0.003536`
- `politeness`: `-0.003604`
- `neurostate_residual`: `0.973931`
- `neurostate_semantic_projection`: `0.226845`

The residual is numerically orthogonal to all four semantic controls:

- residual vs certainty: approximately `0`
- residual vs sentiment: approximately `0`
- residual vs verbosity: approximately `0`
- residual vs politeness: approximately `0`

## Behavioral Pattern

Original NeuroState:

- `proceed_hesitate`: `+0.112664`, p `0.000002`
- `certainty_uncertainty`: `+0.144274`, p `0.000002`
- `positive_negative`: `+0.041264`, p `0.000639`
- `polite_blunt`: `-0.078692`, p `0.000044`
- `verbose_concise`: near zero, p `0.972401`

NeuroState residual:

- `proceed_hesitate`: `+0.149147`, p `0.000002`
- `certainty_uncertainty`: `+0.125903`, p `0.000002`
- `polite_blunt`: `-0.107036`, p `0.000006`
- `positive_negative`: weaker, p `0.061184`
- `verbose_concise`: not significant, p `0.386118`

Semantic projection:

- strongly resembles the learned control basis
- reverses the proceed effect: `proceed_hesitate` slope `-0.144180`
- increases ordinary semantic signals such as certainty, positivity, and
  politeness

## Interpretation

The robust response-boundary NeuroState effect is carried primarily by the
orthogonal residual, not by the ordinary semantic control projection.

The residual preserves the most important previously observed effect:
increased proceed-over-hesitate contrast at the official Qwen chat response
boundary. The semantic projection moves in the opposite direction on that same
contrast.

This supports the interpretation that NeuroState is not merely certainty,
sentiment, verbosity, or politeness steering. It appears to encode a distinct
boundary-sensitive activation direction.

## Natural Next Experiment

v0.17 should test whether this residual direction replicates under independent
prompt and semantic-control training sets.

Recommended design:

- learn controls on one prompt split
- compute NeuroState residual against those controls
- evaluate on a disjoint prompt split
- compare against the original NeuroState direction and fresh random controls
- keep official Qwen chat template, layer 13, position `-1`

