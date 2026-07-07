from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "neurostate_qwen_causal_intervention_v0_5_colab.ipynb"
SOURCE_FILES = ("intervention_core.py", "build_direction.py", "run_intervention.py", "analyze_intervention.py")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown(
        "# NeuroState Qwen Causal Intervention v0.5\n\n"
        "Builds a Qwen2.5-0.5B proceed/hesitate-specific direction, then compares "
        "early layer 7 with the strongest v0.4 site at layer 13.\n\n"
        "Select a T4 GPU runtime, then run all cells."
    ),
    code("!pip -q install 'torch>=2.2' 'transformers>=4.40' 'accelerate>=0.28'\n"),
    code("from pathlib import Path\nWORK=Path('/content/neurostate_qwen_causal_intervention_v0_5')\nWORK.mkdir(parents=True,exist_ok=True)\n"),
]
for name in SOURCE_FILES:
    content = (ROOT / name).read_text(encoding="utf-8")
    cells.append(code(f"(WORK/{name!r}).write_text({content!r},encoding='utf-8')\n"))
cells.extend([
    markdown("## Build v0.5 Direction"),
    code(
        "!cd /content/neurostate_qwen_causal_intervention_v0_5 && "
        "python build_direction.py --target-layer 13 "
        "--output qwen_proceed_hesitate_direction_v0_5.json\n"
    ),
    markdown("## Layer 7 vs 13 Check"),
    code(
        "!cd /content/neurostate_qwen_causal_intervention_v0_5 && "
        "python run_intervention.py --direction-json qwen_proceed_hesitate_direction_v0_5.json "
        "--mode layer_scan --layers 7,13 --prompt-count 8 --random-count 15 "
        "--contrast-bank proceed_hesitate "
        "--output-dir outputs_layer_compare\n"
    ),
    code(
        "!cd /content/neurostate_qwen_causal_intervention_v0_5 && "
        "python analyze_intervention.py --runs outputs_layer_compare/intervention_layer_scan.jsonl "
        "--contrast-bank proceed_hesitate "
        "--output-dir analysis_layer_compare\n"
    ),
    code("from IPython.display import Markdown,display\ndisplay(Markdown((WORK/'analysis_layer_compare'/'causal_report.md').read_text()))\n"),
    markdown("## Optional Layer 13 Full Run\n\nRun this if layer 13 remains stronger than layer 7."),
    code(
        "# !cd /content/neurostate_qwen_causal_intervention_v0_5 && python run_intervention.py --direction-json qwen_proceed_hesitate_direction_v0_5.json --mode full --layers 13 --contrast-bank proceed_hesitate --output-dir outputs_full\n"
        "# !cd /content/neurostate_qwen_causal_intervention_v0_5 && python analyze_intervention.py --runs outputs_full/intervention_full.jsonl --contrast-bank proceed_hesitate --output-dir analysis_full\n"
    ),
    code(
        "import shutil\nfrom google.colab import files\n"
        "archive=shutil.make_archive('/content/neurostate_qwen_causal_intervention_v0_5_results','zip',WORK)\n"
        "files.download(archive)\n"
    ),
])
notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
