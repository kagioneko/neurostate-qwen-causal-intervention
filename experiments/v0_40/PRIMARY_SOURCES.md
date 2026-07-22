# Primary Sources and Local Evidence

## External primary sources

### Refusal direction

- Andy Arditi et al. (2024), **Refusal in Language Models Is Mediated by a Single Direction**
  Paper: https://arxiv.org/abs/2406.11717
  Official code: https://github.com/andyrdt/refusal_direction

The paper reports a model-local residual-stream direction whose removal suppresses refusal and whose addition elicits refusal. The current NeuroState audit does not reproduce the full direction-selection pipeline; it first uses a harmful-minus-harmless activation contrast as a geometry proxy, then tests only benign holdout behavior.

## Local primary evidence

- `outputs/anchored_3axis_results/`: downloaded raw result ZIPs.
- `source_results/generation_scan_v0_29.jsonl`: Qwen generation source rows.
- `analysis_pairwise_review/`: pairwise and blind-review artifacts.
- `colab_neurostate_3axis.py`: anchored v2 three-axis experiment.
- `colab_approach_refusal_holdout_audit.py`: holdout and refusal-proxy audit.
- `NEUROSTATE_3AXIS_RESULTS_JA.md`: consolidated numeric result table.

## Interpretation boundary

- Prompt role-play is not evidence of a corresponding internal variable.
- Activation change is not evidence of semantic specificity without random controls.
- Orthogonal vectors are not necessarily behaviorally independent.
- Anchored v2 is post-hoc relative to the earlier success condition.
- Cross-model slope magnitudes are not directly comparable.
