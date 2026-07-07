from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "neurostate_qwen_causal_intervention_v0_6_colab.ipynb"
SOURCE_FILES = (
    "intervention_core.py",
    "run_intervention.py",
    "analyze_intervention.py",
    "run_position_scan.py",
    "analyze_position_scan.py",
)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown(
        "# NeuroState Qwen Causal Intervention v0.6\n\n"
        "Scans token positions at the layer-13 intervention site using the stronger "
        "inherited NeuroState direction from v0.4.\n\n"
        "Select a T4 GPU runtime, then run all cells."
    ),
    code("!pip -q install 'torch>=2.2' 'transformers>=4.40' 'accelerate>=0.28'\n"),
    code("from pathlib import Path\nWORK=Path('/content/neurostate_qwen_causal_intervention_v0_6')\nWORK.mkdir(parents=True,exist_ok=True)\n"),
]
for name in SOURCE_FILES:
    content = (ROOT / name).read_text(encoding="utf-8")
    cells.append(code(f"(WORK/{name!r}).write_text({content!r},encoding='utf-8')\n"))
direction = (ROOT / "qwen_middle_prompt_direction.json").read_text(encoding="utf-8")
cells.append(code(f"(WORK/'qwen_middle_prompt_direction.json').write_text({direction!r},encoding='utf-8')\n"))
cells.extend([
    markdown("## Token Position Scan"),
    code(
        "!cd /content/neurostate_qwen_causal_intervention_v0_6 && "
        "python run_position_scan.py --direction-json qwen_middle_prompt_direction.json "
        "--target-layer 13 --positions=-1,-2,-4,-8 --prompt-count 8 --random-count 15 "
        "--contrast-bank proceed_hesitate "
        "--output-dir outputs_position_scan\n"
    ),
    code(
        "!cd /content/neurostate_qwen_causal_intervention_v0_6 && "
        "python analyze_position_scan.py --runs outputs_position_scan/position_scan.jsonl "
        "--contrast-bank proceed_hesitate "
        "--output-dir analysis_position_scan\n"
    ),
    code("from IPython.display import Markdown,display\ndisplay(Markdown((WORK/'analysis_position_scan'/'position_report.md').read_text()))\n"),
    markdown("## Optional Last-Token Full Baseline\n\nRun only if you need to reproduce the v0.4 layer-13 full baseline in the same ZIP."),
    code(
        "# !cd /content/neurostate_qwen_causal_intervention_v0_6 && python run_intervention.py --direction-json qwen_middle_prompt_direction.json --mode full --layers 13 --contrast-bank proceed_hesitate --output-dir outputs_full\n"
        "# !cd /content/neurostate_qwen_causal_intervention_v0_6 && python analyze_intervention.py --runs outputs_full/intervention_full.jsonl --contrast-bank proceed_hesitate --output-dir analysis_full\n"
    ),
    code(
        "import shutil\nfrom google.colab import files\n"
        "archive=shutil.make_archive('/content/neurostate_qwen_causal_intervention_v0_6_results','zip',WORK)\n"
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
