# NeuroState Qwen Causal Intervention v0.7

This is a dedicated confirmatory notebook for the v0.6 token-position finding.

## Primary test

- Use the inherited NeuroState direction from v0.4.
- Fix decoder layer to `13`.
- Confirm token positions `-8`, `-2`, and `-1`.
- Use `prompt-count=20`.
- Use `random-count=63`.
- Analyze the `proceed_hesitate` contrast.

The goal is to check whether the weak but highly specific opposite-sign effects
at `-8` and `-2` survive a full-size random-control run, while also reproducing
the strong final-token effect at `-1`.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_7_position_confirm_colab.ipynb`,
select a T4 GPU runtime, and run all cells.

Expected outputs:

- `analysis_position_confirm/position_report.md`
- `analysis_position_confirm/position_summary.csv`
- `analysis_position_confirm/position_summary.json`
