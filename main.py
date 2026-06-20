"""End-to-end demo: run the 3-phase pipeline and evaluate performance predictions."""
import argparse
from glob import glob
import json
from pathlib import Path
from tqdm import tqdm

from src.data_loader import load_all, project_views
from src.config import ID_COL
from src.rag import ProjectRetriever
from src.graph import SelectionPipeline
from src.evaluate import build_outcome_labels, compare, normalize_project_id, summarize_results
from src.prompt_config import ITEM_COLS


def _year_from_value(value) -> int | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raw = str(value).strip()
        return int(raw[:4]) if len(raw) >= 4 and raw[:4].isdigit() else None


def _filter_target_end_year(df, target_end_year: int):
    if "종료연도" in df.columns:
        years = df["종료연도"].map(_year_from_value)
    elif "project_end_year" in df.columns:
        years = df["project_end_year"].map(_year_from_value)
    elif "총연구기간-종료연월일" in df.columns:
        years = df["총연구기간-종료연월일"].map(_year_from_value)
    else:
        raise ValueError("종료연도 필터에 사용할 컬럼을 찾지 못했습니다.")
    return df[years == target_end_year].copy()


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


def _load_completed_ids_from_patterns(patterns: list[str]) -> set[str]:
    """Load completed project ids from JSONL files or glob patterns."""
    completed_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for pattern in patterns:
        matches = glob(pattern) or [pattern]
        for match in matches:
            path = Path(match)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if not path.exists():
                print(f"[skip] No result file matched: {pattern}")
                continue
            _, ids = _load_existing_results(path)
            completed_ids.update(ids)
    return completed_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="샘플 과제 수")
    ap.add_argument("--all", action="store_true", help="projects_labeled_only.csv 전체 과제 실행")
    ap.add_argument("--dataset", type=str, default=None, help="실행할 CSV/XLSX/Parquet 데이터셋 경로")
    ap.add_argument("--out", type=str, default="results.jsonl")
    ap.add_argument("--resume", action="store_true", help="기존 JSONL 결과를 보존하고 완료 과제를 건너뜀")
    ap.add_argument(
        "--skip-existing",
        action="append",
        default=[],
        help="완료 처리할 기존 JSONL 결과 파일 또는 glob 패턴. 여러 번 지정 가능",
    )
    ap.add_argument("--no-validation", action="store_true", help="성과 라벨이 없는 데이터셋 실행: validation/metrics 비움")
    ap.add_argument("--target-end-year", type=int, default=None, help="평가 대상 과제를 특정 종료연도로 제한")
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--shard-count", type=int, default=1, help="전체 작업을 나눌 shard 수")
    ap.add_argument("--shard-index", type=int, default=0, help="현재 프로세스가 처리할 shard 번호(0부터 시작)")
    args = ap.parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")

    print(f"[1/4] Loading dataset …")
    df_all = load_all(dataset_path=args.dataset)
    df_target = df_all
    if args.target_end_year is not None:
        df_target = _filter_target_end_year(df_all, args.target_end_year)
        print(
            f"[filter] Target projects with 종료연도={args.target_end_year}: "
            f"{len(df_target)} / {len(df_all)}"
        )

    if args.no_validation:
        print(f"[2/4] Skipping outcome labels; validation disabled …")
        labels = {}
    else:
        print(f"[2/4] Building paper-performance outcome labels …")
        labels = build_outcome_labels(df_all)

    retriever = None
    if not args.no_rag:
        print(f"[3/4] Building internal RAG index …")
        rag_cols = list(dict.fromkeys(c for cols in ITEM_COLS.values() for c in cols))
        retriever = ProjectRetriever(df_all, {"공통": rag_cols})

    if df_target.empty:
        raise ValueError("평가 대상 과제가 없습니다.")

    shard_label = (
        f" shard {args.shard_index}/{args.shard_count}"
        if args.shard_count > 1 else ""
    )
    run_label = ("all" if args.all else str(args.n)) + shard_label
    dataset_label = args.dataset or "projects_labeled_only.csv"
    if args.target_end_year is not None:
        dataset_label += f" (target 종료연도={args.target_end_year}; RAG pool=all eligible historical rows)"
    print(f"[4/4] Running pipeline on {run_label} projects from {dataset_label} …")
    pipeline = SelectionPipeline(retriever=retriever)
    if args.all:
        sample = df_target
    else:
        sample_n = min(args.n, len(df_target))
        sample = df_target.sample(n=sample_n, random_state=42)
    if args.shard_count > 1:
        sample = sample[
            sample["__dataset_row_id"].map(lambda value: int(value) % args.shard_count == args.shard_index)
        ]
        print(
            f"[shard] Selected {len(sample)} projects for shard "
            f"{args.shard_index}/{args.shard_count}."
        )
    sample = sample.reset_index(drop=True)

    out_path = Path(args.out)
    results, completed_ids = _load_existing_results(out_path) if args.resume else ([], set())
    if args.skip_existing:
        skip_ids = _load_completed_ids_from_patterns(args.skip_existing)
        before = len(completed_ids)
        completed_ids.update(skip_ids)
        print(
            f"[skip] Loaded {len(skip_ids)} completed ids from --skip-existing; "
            f"{len(completed_ids) - before} new for this shard."
        )
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
            if args.no_validation:
                result["validation"] = {}
            else:
                outcome = labels.get(normalize_project_id(project["id"]))
                result["validation"] = compare(result["verdict"], outcome)
            fo.write(json.dumps(result, ensure_ascii=False) + "\n")
            fo.flush()  # 실시간 tail 을 위해 매 건 flush
            results.append(result)

    metrics_path = f"{out_path}.metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fo:
        metrics = {"n_results": len(results), "criteria": {}, "main": {}} if args.no_validation else summarize_results(results)
        json.dump(metrics, fo, ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}")
    print(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
