import json
import re
from datetime import datetime
from pathlib import Path


def latest_file(pattern: str, root: Path):
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def select_old40_train_log(logs_root: Path):
    candidate = logs_root / "enhanced_attention_unet" / "20260411_200313" / "train.log"
    return candidate if candidate.exists() else None


def extract_timestamp(name: str):
    match = re.search(r"(\d{8}_\d{6})", name)
    return match.group(1) if match else ""


def select_old40_eval(results_root: Path):
    summaries = sorted(results_root.glob("compare_old_on_*/evaluation_summary_*.json"))
    csvs = sorted(results_root.glob("compare_old_on_*/per_sample_results_*.csv"))
    if not summaries:
        return None, None, None

    by_ts_csv = {extract_timestamp(p.name): p for p in csvs}
    summary = summaries[-1]
    ts = extract_timestamp(summary.name)
    csv = by_ts_csv.get(ts)
    return summary.parent, summary, csv


def main() -> None:
    repo = Path(__file__).resolve().parent
    logs = repo / "logs"
    results = repo / "results"

    train_log = select_old40_train_log(logs) if logs.exists() else None
    eval_dir, eval_json, per_sample_csv = select_old40_eval(results) if results.exists() else (None, None, None)

    if train_log is None:
        train_log = latest_file("**/train.log", logs) if logs.exists() else None
    if eval_json is None:
        eval_json = latest_file("**/evaluation_summary_*.json", results) if results.exists() else None
        eval_dir = eval_json.parent if eval_json else None
    if per_sample_csv is None:
        per_sample_csv = latest_file("**/per_sample_results_*.csv", results) if results.exists() else None

    viz_dir = eval_dir / "visualizations" if eval_dir else None
    viz = sorted(viz_dir.glob("*.png")) if viz_dir and viz_dir.exists() else []

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_log": str(train_log.relative_to(repo)) if train_log else None,
        "evaluation_dir": str(eval_dir.relative_to(repo)) if eval_dir else None,
        "evaluation_summary": str(eval_json.relative_to(repo)) if eval_json else None,
        "per_sample_csv": str(per_sample_csv.relative_to(repo)) if per_sample_csv else None,
        "visualizations_count": len(viz),
        "visualizations_examples": [str(p.relative_to(repo)) for p in viz[:12]],
    }

    out = repo / "docs" / "ARTIFACT_INDEX.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Artifact index written: {out}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
