from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "neurostate_qwen_causal_intervention_v0_9_token_inspect_colab.ipynb"
SOURCE_FILES = ("intervention_core.py", "run_intervention.py", "inspect_prompt_tokens.py")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown(
        "# NeuroState Qwen Causal Intervention v0.9\n\n"
        "Inspects which tokenizer tokens correspond to the v0.8 token-position map. "
        "This notebook does not load the language model; it only loads the tokenizer."
    ),
    code("!pip -q install 'transformers>=4.40'\n"),
    code("from pathlib import Path\nWORK=Path('/content/neurostate_qwen_causal_intervention_v0_9')\nWORK.mkdir(parents=True,exist_ok=True)\n"),
]
for name in SOURCE_FILES:
    content = (ROOT / name).read_text(encoding="utf-8")
    cells.append(code(f"(WORK/{name!r}).write_text({content!r},encoding='utf-8')\n"))
cells.extend(
    [
        markdown("## Inspect Position Tokens"),
        code(
            "!cd /content/neurostate_qwen_causal_intervention_v0_9 && "
            "python inspect_prompt_tokens.py --positions=-12,-10,-8,-6,-4,-2,-1 "
            "--prompt-count 12 --output-dir token_inspect\n"
        ),
        code(
            "from IPython.display import Markdown,display\n"
            "report=WORK/'token_inspect'/'prompt_position_tokens.md'\n"
            "display(Markdown(report.read_text()))\n"
        ),
        code(
            "import shutil\nfrom google.colab import files\n"
            "archive=shutil.make_archive('/content/neurostate_qwen_causal_intervention_v0_9_results','zip',WORK)\n"
            "files.download(archive)\n"
        ),
    ]
)
notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
