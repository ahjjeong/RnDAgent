"""End-to-end demo: run the 3-phase pipeline and evaluate performance predictions."""
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.data_loader import load_all, project_views
from src.config import ID_COL
from src.rag import ProjectRetriever
from src.graph import SelectionPipeline
from src.evaluate import build_outcome_labels, compare, normalize_project_id, summarize_results
from src.prompt_config import ITEM_COLS


def _load_existing_results(path: Path) -> tuple[list[dict], set[str]]:
    """Read completed JSONL rows so long full runs can resume after timeout."""
    if not path.exists():
        return [], set()

    results: list[dict] = []
    completed_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fi:
        for line_no, line in enumerate(fi, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] Skipping malformed line {line_no} in {path}")
                continue
            results.append(result)
            pid = normalize_project_id(result.get("project_id"))
            if pid:
                completed_ids.add(pid)
    return results, completed_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="샘플 과제 수")
    ap.add_argument("--all", action="store_true", help="projects_labeled_only.csv 전체 과제 실행")
    ap.add_argument("--out", type=str, default="results.jsonl")
    ap.add_argument("--resume", action="store_true", help="기존 JSONL 결과를 보존하고 완료 과제를 건너뜀")
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()

    print(f"[1/4] Loading dataset …")
    df_all = load_all()                   # projects_labeled_only.csv 전체
    df_target = df_all

    print(f"[2/4] Building paper-performance outcome labels …")
    labels = build_outcome_labels(df_all)

    retriever = None
    if not args.no_rag:
        print(f"[3/4] Building internal RAG index …")
        rag_cols = list(dict.fromkeys(c for cols in ITEM_COLS.values() for c in cols))
        retriever = ProjectRetriever(df_all, {"공통": rag_cols})

    if df_target.empty:
        raise ValueError("평가 대상 과제가 없습니다.")

    run_label = "all" if args.all else str(args.n)
    print(f"[4/4] Running pipeline on {run_label} projects from projects_labeled_only.csv …")
    pipeline = SelectionPipeline(retriever=retriever)
    if args.all:
        sample = df_target.reset_index(drop=True)
    else:
        sample_n = min(args.n, len(df_target))
        sample = df_target.sample(n=sample_n, random_state=42).reset_index(drop=True)

    out_path = Path(args.out)
    results, completed_ids = _load_existing_results(out_path) if args.resume else ([], set())
    if completed_ids:
        sample = sample[
            ~sample[ID_COL].map(lambda value: normalize_project_id(value) in completed_ids)
        ].reset_index(drop=True)
        print(f"[resume] Loaded {len(results)} existing results; {len(sample)} projects remaining.")

    mode = "a" if args.resume and out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8") as fo:
        for _, row in tqdm(sample.iterrows(), total=len(sample)):
            project = project_views(row)
            result = pipeline.run(project)
            outcome = labels.get(normalize_project_id(project["id"]))
            result["validation"] = compare(result["verdict"], outcome)
            fo.write(json.dumps(result, ensure_ascii=False) + "\n")
            fo.flush()  # 실시간 tail 을 위해 매 건 flush
            results.append(result)

    metrics_path = f"{out_path}.metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fo:
        json.dump(summarize_results(results), fo, ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}")
    print(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
