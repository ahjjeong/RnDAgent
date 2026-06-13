"""results.jsonl 를 사람이 읽기 좋은 형태로 출력.

사용법:
    python view_results.py                       # 전체 결과 요약
    python view_results.py --file results.jsonl  # 파일 지정
    python view_results.py --idx 0               # 특정 과제만 상세 출력
    python view_results.py --phase 1             # Phase 1 만
    python view_results.py --raw                 # JSON 원본도 같이
"""
import argparse
import json
from pathlib import Path


SEP = "=" * 80
SUB = "-" * 80


def fmt_eval(e: dict) -> str:
    lines = [f"  ▸ {e.get('agent','?')}  | score = {e.get('score','?')}"]
    if "reasoning" in e:
        lines.append(f"    reasoning : {e['reasoning']}")
    if e.get("strengths"):
        lines.append(f"    strengths : {e['strengths']}")
    if e.get("weaknesses"):
        lines.append(f"    weaknesses: {e['weaknesses']}")
    if "_raw" in e:
        lines.append(f"    [JSON 파싱 실패 — raw 응답]\n    {e['_raw'][:400]}")
    return "\n".join(lines)


def show(r: dict, idx: int, phase_filter: int | None, raw: bool):
    print(SEP)
    print(f"[#{idx}] {r.get('title','?')}")
    print(f"       id = {r.get('project_id','?')}")
    v = r.get("validation", {})
    if v:
        mark = "✓" if v.get("aligned") else "✗"
        print(f"       decision={v.get('decision')}  rank={v.get('priority_rank')}  "
              f"outcome={v.get('outcome_grade')}  aligned={mark}")
    print(SEP)

    if phase_filter in (None, 1):
        print("\n▣ Phase 1 — 독립 평가")
        print(SUB)
        for e in r.get("phase1", []):
            print(fmt_eval(e))
            print()

    if phase_filter in (None, 2):
        print("▣ Phase 2 — Coordinator 쟁점")
        print(SUB)
        print(r.get("coordinator_issues", "(없음)"))
        print()
        print("▣ Phase 2 — Round 2 반론")
        print(SUB)
        for rb in r.get("phase2_rebuttals", []):
            print(f"  ▸ {rb.get('agent','?')}")
            print(f"    {rb.get('response','')}")
            print()

    if phase_filter in (None, 3):
        print("▣ Phase 3 — Moderator 최종 판정")
        print(SUB)
        verdict = r.get("verdict", {})
        for k, val in verdict.items():
            print(f"  {k}: {val}")
        print()

    if raw:
        print("▣ RAW JSON")
        print(SUB)
        print(json.dumps(r, ensure_ascii=False, indent=2))


def summarize(results: list[dict]):
    print(SEP)
    print(f"Summary  (총 {len(results)} 건)")
    print(SEP)
    print(f"{'idx':<4} {'decision':<12} {'rank':<6} {'outcome':<8} {'align':<6} title")
    for i, r in enumerate(results):
        v = r.get("validation", {})
        print(f"{i:<4} "
              f"{str(v.get('decision','?'))[:11]:<12} "
              f"{str(v.get('priority_rank','?'))[:5]:<6} "
              f"{str(v.get('outcome_grade','?'))[:7]:<8} "
              f"{'✓' if v.get('aligned') else '✗':<6} "
              f"{str(r.get('title',''))[:60]}")
    n = len(results)
    if n:
        aligned = sum(1 for r in results if r.get("validation", {}).get("aligned"))
        print(SUB)
        print(f"alignment rate: {aligned}/{n} = {aligned/n:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results.jsonl")
    ap.add_argument("--idx", type=int, default=None,
                    help="특정 결과만 상세 출력 (0-based)")
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"파일 없음: {path}")
        return

    results = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.idx is None:
        summarize(results)
        print()
        for i, r in enumerate(results):
            show(r, i, args.phase, args.raw)
    else:
        show(results[args.idx], args.idx, args.phase, args.raw)


if __name__ == "__main__":
    main()
