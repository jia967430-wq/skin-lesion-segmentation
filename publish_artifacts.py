import json
from datetime import datetime
from pathlib import Path


def latest_file(pattern: str, root: Path):
    files = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def main() -> None:
    repo = Path(__file__).resolve().parent
    logs = repo / "logs"
    results = repo / "results"

    latest_train_log = latest_file("**/train.log", logs) if logs.exists() else None
    latest_eval_json = latest_file("evaluation_summary_*.json", results) if results.exists() else None
    latest_csv = latest_file("per_sample_results_*.csv", results) if results.exists() else None
    viz = sorted((results / "visualizations").glob("*.png")) if (results / "visualizations").exists() else []

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train_log": str(latest_train_log.relative_to(repo)) if latest_train_log else None,
        "evaluation_summary": str(latest_eval_json.relative_to(repo)) if latest_eval_json else None,
        "per_sample_csv": str(latest_csv.relative_to(repo)) if latest_csv else None,
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
