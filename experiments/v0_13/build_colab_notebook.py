from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "neurostate_qwen_causal_intervention_v0_13_independent_seed_colab.ipynb"
SOURCE_FILES = (
    "intervention_core.py",
    "run_intervention.py",
    "run_position_scan.py",
    "run_chat_template_scan.py",
    "analyze_format_position_scan.py",
    "analyze_intervention.py",
)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown(
        "# NeuroState Qwen Causal Intervention v0.13\n\n"
        "Independent random-control seed replication of the official Qwen chat-template result."
    ),
    code("!pip -q install 'torch>=2.2' 'transformers>=4.40' 'accelerate>=0.28'\n"),
    code("from pathlib import Path\nWORK=Path('/content/neurostate_qwen_causal_intervention_v0_13')\nWORK.mkdir(parents=True,exist_ok=True)\n"),
]
for name in SOURCE_FILES:
    content = (ROOT / name).read_text(encoding="utf-8")
    cells.append(code(f"(WORK/{name!r}).write_text({content!r},encoding='utf-8')\n"))
direction = (ROOT / "qwen_middle_prompt_direction.json").read_text(encoding="utf-8")
cells.append(code(f"(WORK/'qwen_middle_prompt_direction.json').write_text({direction!r},encoding='utf-8')\n"))
cells.extend(
    [
        markdown("## Independent-Seed Replication"),
        code(
            "!cd /content/neurostate_qwen_causal_intervention_v0_13 && "
            "python run_chat_template_scan.py --direction-json qwen_middle_prompt_direction.json "
            "--target-layer 13 --token-position=-1 --prompt-count 20 --random-count 63 "
            "--random-seed 20260713 --contrast-bank proceed_hesitate "
            "--output-dir outputs_independent_seed\n"
        ),
        code(
            "!cd /content/neurostate_qwen_causal_intervention_v0_13 && "
            "python analyze_format_position_scan.py --runs outputs_independent_seed/format_position_scan.jsonl "
            "--contrast-bank proceed_hesitate --output-dir analysis_independent_seed\n"
        ),
        code(
            "from IPython.display import Markdown,display\n"
            "report=WORK/'analysis_independent_seed'/'format_position_report.md'\n"
            "display(Markdown(report.read_text()))\n"
        ),
        code(
            "import json\n"
            "first=json.loads((WORK/'outputs_independent_seed'/'format_position_scan.jsonl').read_text().splitlines()[0])\n"
            "print('Boundary:', repr(first['boundary_token']), 'id=', first['boundary_token_id'], 'seed=', first['random_seed'])\n"
        ),
        code(
            "import shutil\nfrom google.colab import files\n"
            "archive=shutil.make_archive('/content/neurostate_qwen_causal_intervention_v0_13_results','zip',WORK)\n"
            "files.download(archive)\n"
        ),
    ]
)
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
