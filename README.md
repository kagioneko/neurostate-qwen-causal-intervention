# NeuroState Qwen Causal Intervention

This repository archives a sequence of small causal intervention experiments on
Qwen-family and comparison-model hidden states.

The core question is whether a NeuroState-derived activation direction affects
LLM behavior as a functional internal control signal, rather than only as an
ordinary surface-semantic direction such as certainty, sentiment, verbosity, or
politeness.

## Repository Layout

- `experiments/v0_1` through `experiments/v0_20`: historical versioned snapshots.
- `experiments/v0_40`: multi-model audit, Qwen3 holdout/refusal audit, vigilance
  follow-up, and the interactive steering demo.
- `articles`: Japanese beginner and publication-oriented explanations.
- Each version is self-contained and includes its own scripts, requirements, and
  Colab notebook when available.
- Later versions include stronger controls and result summaries.

## Main Result Trail

- `v0_15`: semantic control baseline setup.
- `v0_16`: decomposes NeuroState into semantic projection and semantic residual.
  The residual remains strongly aligned with the proceed-over-hesitate effect.
- `v0_17`: disjoint train/eval split confirms the residual effect generalizes
  beyond the prompts used to learn the semantic controls.
- `v0_18`: reverse split reproduces the residual effect.
- `v0_19`: layer scan shows the strongest residual effect at Qwen layer 13.
- `v0_20`: position scan setup for layer 13 around the chat response boundary.
- `v0_40`: approach survives a new Qwen3 holdout; refusal overlap is strongly
  layer-dependent; vigilance remains indistinguishable from random controls on
  a false-premise single-token follow-up.

Key summaries:

- `experiments/v0_16/RESULT_SUMMARY.md`
- `experiments/v0_17/RESULT_SUMMARY.md`
- `experiments/v0_18/RESULT_SUMMARY.md`
- `experiments/v0_19/RESULT_SUMMARY.md`
- `experiments/v0_20/RESULT_SUMMARY.md`
- `experiments/v0_40/README.md`
- `experiments/v0_40/NEUROSTATE_3AXIS_RESULTS_JA.md`
- `articles/NOTION_ARTICLE_NEUROSTATE_BEGINNER_JA.md`

## Current Interpretation

The updated evidence supports a narrower interpretation. Descriptive
NeuroState dimensions can organize prompts, but they were not recovered as six
independent causal controls. Approach is the clearest reproducible direction;
vigilance did not separate from random controls under the tested measures.

This does not establish that the direction is a human-like emotion state. A more
careful interpretation is that approach is one model- and layer-dependent
direction that can causally bias continuation behavior under specified tests.

## Related Context

`experiments/v0_19/RELATED_J_SPACE_NOTE.md` notes conceptual overlap with
Anthropic's J-space / J-lens discussion. This repository does not implement that
method. The connection is interpretive: both lines of work care about whether
latent internal representations have functional effects on model outputs.

## Running A Version

Open the Colab notebook in a chosen experiment directory, select a GPU runtime,
and run all cells. For example:

```text
experiments/v0_20/neurostate_qwen_causal_intervention_v0_20_residual_position_scan_colab.ipynb
```

For local tests in a version directory:

```powershell
$env:PYTHONPATH='.'
pytest tests
```

## Notes

- No model weights are stored in this repository.
- Generated caches and large local output folders should stay untracked.
- The experiment snapshots intentionally preserve historical scripts rather than
  refactoring them into a single shared package.
