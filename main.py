"""End-to-end demo: load data, sample N projects, run the 3-phase pipeline,
and compare verdicts with proxy outcome labels."""
import argparse
import json
from tqdm import tqdm

from src.data_loader import load_all, project_views
from src.config import AGENT1_COLS, AGENT2_COLS, AGENT3_COLS
from src.rag import ProjectRetriever
from src.graph import SelectionPipeline
from src.evaluate import build_outcome_labels, compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="샘플 과제 수")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--out", type=str, default="results.jsonl")
    ap.add_argument("--no-rag", action="store_true")
    args = ap.parse_args()

    print(f"[1/4] Loading dataset …")
    df_all = load_all()                   # 전 연도 (성과 라벨용)
    df_target = load_all(years=[args.year])  # 평가 대상 연도

    print(f"[2/4] Building outcome labels …")
    labels = build_outcome_labels(df_all)

    retriever = None
    if not args.no_rag:
        print(f"[3/4] Building RAG indices …")
        retriever = ProjectRetriever(
            df_all,
            {"agent1": AGENT1_COLS, "agent2": AGENT2_COLS, "agent3": AGENT3_COLS},
        )

    print(f"[4/4] Running pipeline on {args.n} projects …")
    pipeline = SelectionPipeline(retriever=retriever)
    sample = df_target.sample(n=args.n, random_state=42).reset_index(drop=True)

    with open(args.out, "w", encoding="utf-8") as fo:
        for _, row in tqdm(sample.iterrows(), total=len(sample)):
            project = project_views(row)
            result = pipeline.run(project)
            grade = labels.get(project["id"], "?")
            result["validation"] = compare(result["verdict"], grade)
            fo.write(json.dumps(result, ensure_ascii=False) + "\n")
            fo.flush()  # 실시간 tail 을 위해 매 건 flush
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
