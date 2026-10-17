"""Build a static copy of the dashboard for hosts with no backend (e.g. a static Hugging Face Space).

    python -m experiments.export_static --out ../agent-reliability-lab-static [--space-readme]

Writes index.html (the same page the API serves) plus data.json holding everything the page asks the API
for: dataset list, summaries, per-trial records, the ablation matrix and the main run's traces.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.analyze import ablation_matrix, summarize
from experiments.runner import RESULTS_DIR, load_done

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

SPACE_README = """---
title: Agent Reliability Lab
emoji: 🧪
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: false
short_description: Fault-injection benchmark for LLM agents
---

# Agent Reliability Lab

Interactive dashboard for a fault-injection benchmark that measures how much of an LLM agent's reliability
comes from the engineering around the model. It shows **recorded results**; nothing runs live.

Source, results and write-up: {repo}
"""


def build_bundle(main: str = "main", ablation: str = "ablation", results_dir: Path = RESULTS_DIR) -> dict:
    main_rows = load_done(results_dir / f"{main}.jsonl")
    abl_rows = load_done(results_dir / f"{ablation}.jsonl")
    if not main_rows:
        raise SystemExit(f"no results for '{main}'")
    traces_path = results_dir / f"{main}.traces.json"
    datasets = []
    for name, rows in ((main, main_rows), (ablation, abl_rows)):
        if rows:
            datasets.append({"name": name, "trials": len(rows),
                             "configs": sorted({r["config"] for r in rows}),
                             "models": sorted({r["model"] for r in rows})})
    return {
        "datasets": datasets,
        "summary": {main: summarize(main_rows)},
        "records": {main: main_rows},
        "ablation": ablation_matrix(main_rows, abl_rows) if abl_rows else None,
        "traces": {main: json.loads(traces_path.read_text()) if traces_path.exists() else {}},
    }


def export(out: Path, space_readme: bool = False,
           repo: str = "https://github.com/vasudhanagraj27/agent-reliability-lab") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(STATIC_DIR / "index.html", out / "index.html")
    (out / "data.json").write_text(json.dumps(build_bundle(), separators=(",", ":")))
    if space_readme:
        (out / "README.md").write_text(SPACE_README.format(repo=repo))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--space-readme", action="store_true", help="also write the Hugging Face Space README")
    args = ap.parse_args()
    out = export(args.out, args.space_readme)
    print(f"wrote {out}/index.html and data.json ({(out / 'data.json').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
