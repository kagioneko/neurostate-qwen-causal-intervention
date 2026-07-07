# NeuroState Qwen Causal Intervention v0.13

Independent random-control seed replication of v0.12.

## Fixed conditions

- Qwen official chat template with `add_generation_prompt=True`
- inherited NeuroState direction
- decoder layer `13`, token position `-1`
- 20 prompts and 63 random controls
- `proceed_hesitate` logit contrast

The only experimental change from v0.12 is the random-control seed,
`20260713`. The seed is stored in every JSONL row.

## Colab

Upload `neurostate_qwen_causal_intervention_v0_13_independent_seed_colab.ipynb`,
select a T4 GPU runtime, and run all cells. The final cell downloads the result
ZIP.
