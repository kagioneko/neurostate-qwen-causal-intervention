# NeuroState Qwen Causal Intervention v0.9

This notebook inspects which tokenizer tokens correspond to the v0.8
token-position map.

## Primary test

- Load the Qwen2.5-0.5B tokenizer only.
- Use the same first 12 prompts as the v0.8 map.
- Inspect positions `-12`, `-10`, `-8`, `-6`, `-4`, `-2`, and `-1`.
- Export a Markdown table and CSV.

Expected outputs:

- `token_inspect/prompt_position_tokens.md`
- `token_inspect/prompt_position_tokens.csv`

This is an interpretation helper, not a causal intervention run.
