# NeuroState Qwen Causal Intervention v0.12

This confirmatory run tests the layer-13 response-boundary effect with Qwen's
official tokenizer chat template instead of a manually written `Assistant:`
prompt.

## Primary test

- Build each prompt with `tokenizer.apply_chat_template()`.
- Set `add_generation_prompt=True`.
- Apply the inherited NeuroState direction at decoder layer `13`.
- Intervene on position `-1`, the final token of the generation boundary.
- Use 20 prompts and 63 orthogonal random control directions.
- Analyze the `proceed_hesitate` logit contrast.
- Save the actual boundary token ID and token text in every result row.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_12_chat_template_confirm_colab.ipynb`,
select a T4 GPU runtime, and run all cells. The final cell downloads the complete
results ZIP.
