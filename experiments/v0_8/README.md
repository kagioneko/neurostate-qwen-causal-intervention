# NeuroState Qwen Causal Intervention v0.8

This is a token-position map around the v0.7 hotspots.

## Primary test

- Use the inherited NeuroState direction from v0.4.
- Fix decoder layer to `13`.
- Scan token positions `-12`, `-10`, `-8`, `-6`, `-4`, `-2`, and `-1`.
- Use `prompt-count=12`.
- Use `random-count=31`.
- Analyze the `proceed_hesitate` contrast.

This is a mapping run, not a final confirmatory run. The goal is to locate the
shape around the specific earlier-position effects at `-8` and `-2`, while also
keeping the strong last-token site `-1` in the same table.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_8_position_map_colab.ipynb`,
select a T4 GPU runtime, and run all cells.

Expected outputs:

- `analysis_position_map/position_report.md`
- `analysis_position_map/position_summary.csv`
- `analysis_position_map/position_summary.json`
