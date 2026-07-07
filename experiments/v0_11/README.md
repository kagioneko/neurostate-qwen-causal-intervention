# NeuroState Qwen Causal Intervention v0.11

This is a confirmatory run for the `User: ...\nAssistant:` response-boundary
effect found in v0.10.

## Primary test

- Use the inherited NeuroState direction from v0.4.
- Fix decoder layer to `13`.
- Use prompt format `User: {prompt}\nAssistant:`.
- Test token position `-1`, the final colon after `Assistant`.
- Use `prompt-count=20`.
- Use `random-count=63`.
- Analyze the `proceed_hesitate` contrast.

The goal is to check whether the `Assistant:` boundary remains stronger and
more specific than the original `Response:` boundary under full-size controls.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_11_user_assistant_confirm_colab.ipynb`,
select a T4 GPU runtime, and run all cells.

Expected outputs:

- `analysis_user_assistant_confirm/format_position_report.md`
- `analysis_user_assistant_confirm/format_position_summary.csv`
- `analysis_user_assistant_confirm/format_position_summary.json`
