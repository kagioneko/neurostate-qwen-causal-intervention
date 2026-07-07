# NeuroState Qwen Causal Intervention v0.10

This experiment tests whether the boundary-position effects found in v0.7-v0.9
are specific to `Response:` or generalize to other response-boundary formats.

## Primary test

- Use the inherited NeuroState direction from v0.4.
- Fix decoder layer to `13`.
- Compare prompt formats:
  - `Task: ...\nResponse:`
  - `Instruction: ...\nAnswer:`
  - `User: ...\nAssistant:`
- Test positions `-2` and `-1`.
- Use `prompt-count=8`.
- Use `random-count=15`.
- Analyze the `proceed_hesitate` contrast.

This is a small format-generalization map. If the sign pattern survives across
formats, the effect is probably tied to response-boundary structure rather than
the literal `Response:` string.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_10_format_position_colab.ipynb`,
select a T4 GPU runtime, and run all cells.

Expected outputs:

- `analysis_format_position/format_position_report.md`
- `analysis_format_position/format_position_summary.csv`
- `analysis_format_position/format_position_summary.json`
