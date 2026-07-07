from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervention_core import linear_slope


def sign_matrix(count: int):
    import numpy as np

    masks = np.arange(1 << count, dtype=np.uint32)[:, None]
    bits = (masks >> np.arange(count, dtype=np.uint32)) & 1
    return np.where(bits == 0, 1.0, -1.0)


def exact_sign_flip_p_numpy(values, signs) -> float:
    import numpy as np

    vector = np.asarray(values, dtype=np.float64)
    observed = abs(vector.mean())
    permuted = np.abs((signs @ vector) / len(vector))
    return float(np.mean(permuted >= observed - 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.runs.read_text(encoding="utf-8").splitlines() if line.strip()]
    direction_names = sorted({row["direction_name"] for row in rows}, key=lambda name: (name != "neurostate", name))
    bank_names = sorted(key.removeprefix("logit_contrast_") for key in rows[0] if key.startswith("logit_contrast_"))
    signs = sign_matrix(len({row["prompt_id"] for row in rows}))
    results = []
    for direction_name in direction_names:
        direction_rows = [row for row in rows if row["direction_name"] == direction_name]
        for bank_name in bank_names:
            slopes = []
            for prompt_id in sorted({row["prompt_id"] for row in direction_rows}):
                prompt_rows = sorted((row for row in direction_rows if row["prompt_id"] == prompt_id), key=lambda row: row["alpha"])
                slopes.append(linear_slope([row["alpha"] for row in prompt_rows], [row[f"logit_contrast_{bank_name}"] for row in prompt_rows]))
            results.append({
                "direction_name": direction_name,
                "measure_bank": bank_name,
                "prompt_count": len(slopes),
                "mean_slope": sum(slopes) / len(slopes),
                "median_slope": sorted(slopes)[len(slopes) // 2],
                "sign_flip_p": exact_sign_flip_p_numpy(slopes, signs),
            })
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"mode": "semantic_control_scan", "geometry": geometry, "results": results}
    (args.output_dir / "semantic_control_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Semantic Direction Control Scan", "", "| direction | measure | mean slope | sign-flip p |", "|---|---|---:|---:|"]
    lines.extend(f"| {r['direction_name']} | {r['measure_bank']} | {r['mean_slope']:.6f} | {r['sign_flip_p']:.6f} |" for r in results)
    lines += ["", "## Cosine with NeuroState", ""]
    lines.extend(f"- `{name}`: {value:.6f}" for name, value in geometry["cosine_with_neurostate"].items())
    (args.output_dir / "semantic_control_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "semantic_control_report.md")


if __name__ == "__main__":
    main()
