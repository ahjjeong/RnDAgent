"""Compare Moderator verdicts against a proxy outcome label.

Note: The raw dataset exposes no explicit 성과 등급. We build a proxy label
from signals that *do* exist across the xlsx files:

  - 신규계속구분 == '계속' in a later year  → 후속 연차 승인 (success proxy)
  - 이전과제고유번호 linkage across years   → follow-up project exists
  - 민간연구비 증가 추이                    → 사업화 관심 증가 proxy

Real 성과 등급(최종평가 S/A/B/C)은 별도 성과 DB 연계가 필요하며,
해당 DB가 제공되면 `build_outcome_labels` 를 교체하세요.
"""
from __future__ import annotations
import pandas as pd
from collections import defaultdict
from .config import ID_COL


def build_outcome_labels(df_all_years: pd.DataFrame) -> dict[str, str]:
    """Return {과제고유번호: 'A'|'B'|'C'|'D'} proxy grade.

    A: follow-up project linked AND 민간연구비 > 0 in later year
    B: follow-up project linked
    C: 단년 종료, 민간연구비 > 0
    D: 단년 종료, 민간연구비 == 0
    """
    per_id = defaultdict(list)
    for _, row in df_all_years.iterrows():
        pid = row.get(ID_COL)
        if pd.isna(pid):
            continue
        per_id[pid].append(row)

    prev_ids = set(df_all_years.get("이전과제고유번호", pd.Series(dtype=object)).dropna())

    labels = {}
    for pid, rows in per_id.items():
        has_followup = pid in prev_ids or len(rows) > 1
        private = sum(float(r.get("민간연구비합계(원)", 0) or 0) for r in rows)
        if has_followup and private > 0:
            labels[pid] = "A"
        elif has_followup:
            labels[pid] = "B"
        elif private > 0:
            labels[pid] = "C"
        else:
            labels[pid] = "D"
    return labels


def compare(verdict: dict, outcome_grade: str) -> dict:
    decision = verdict.get("decision", "")
    rank = verdict.get("priority_rank", "")
    # simple alignment: 선정+A/B = match; 비선정+C/D = match; else partial
    selected = decision in ("선정", "조건부 선정")
    good_outcome = outcome_grade in ("A", "B")
    aligned = selected == good_outcome
    return {
        "decision": decision,
        "priority_rank": rank,
        "outcome_grade": outcome_grade,
        "aligned": aligned,
    }
