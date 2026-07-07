from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "neurostate_qwen_causal_intervention_v0_17_residual_replication_colab.ipynb"
SOURCE_FILES = (
    "intervention_core.py",
    "run_intervention.py",
    "run_position_scan.py",
    "run_chat_template_scan.py",
    "run_semantic_control_scan.py",
    "analyze_semantic_control_scan.py",
)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("# NeuroState Qwen Causal Intervention v0.17\n\nDisjoint-prompt residual replication at the official Qwen response boundary."),
    code("!pip -q install 'torch>=2.2' 'transformers>=4.40' 'accelerate>=0.28'\n"),
    code("from pathlib import Path\nWORK=Path('/content/neurostate_qwen_causal_intervention_v0_17')\nWORK.mkdir(parents=True,exist_ok=True)\n"),
]
for name in SOURCE_FILES:
    content = (ROOT / name).read_text(encoding="utf-8")
    cells.append(code(f"(WORK/{name!r}).write_text({content!r},encoding='utf-8')\n"))
direction = (ROOT / "qwen_middle_prompt_direction.json").read_text(encoding="utf-8")
cells.append(code(f"(WORK/'qwen_middle_prompt_direction.json').write_text({direction!r},encoding='utf-8')\n"))
cells.extend([
    markdown("## Learn Controls and Intervene"),
    code(
        "!cd /content/neurostate_qwen_causal_intervention_v0_17 && python run_semantic_control_scan.py "
        "--direction-json qwen_middle_prompt_direction.json --source-layer 11 --target-layer 13 "
        "--token-position=-1 --train-start 0 --train-count 10 --eval-start 10 --eval-count 10 "
        "--random-count 15 --random-seed 20260717 --output-dir outputs_residual_replication\n"
    ),
    code(
        "!cd /content/neurostate_qwen_causal_intervention_v0_17 && python analyze_semantic_control_scan.py "
        "--runs outputs_residual_replication/semantic_control_scan.jsonl "
        "--geometry outputs_residual_replication/direction_geometry.json --output-dir analysis_residual_replication\n"
    ),
    code("from IPython.display import Markdown,display\ndisplay(Markdown((WORK/'analysis_residual_replication'/'semantic_control_report.md').read_text()))\n"),
    code(
        "import shutil\nfrom google.colab import files\n"
        "archive=shutil.make_archive('/content/neurostate_qwen_causal_intervention_v0_17_results','zip',WORK)\n"
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
