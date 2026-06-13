"""계속과제 사전 성과 조회.

신규계속구분 == '계속'인 과제에 대해,
평가 대상 연도보다 이전 연도의 성과(논문·특허·사업화·기술료)를
dataset/성과/*.xlsx 에서 찾아 요약 문자열로 반환한다.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import pandas as pd

OUTCOME_DIR = Path(__file__).resolve().parent.parent / "dataset" / "성과"

# 성과 유형별 파일 접미사 → 추출할 컬럼
OUTCOME_SPECS = {
    "SCI논문": {
        "suffix": "SCI(E)논문.xlsx",
        "cols": ["성과발생년도", "과제고유번호", "논문명", "학술지명", "SCI여부(입력시)"],
        "label": "SCI(E) 논문",
    },
    "특허": {
        "suffix": "국내(외)특허.xlsx",
        "cols": ["성과발생년도", "과제고유번호", "발명의 명칭", "출원/등록 구분", "출원/등록 국가"],
        "label": "특허",
    },
    "사업화": {
        "suffix": "사업화.xlsx",
        "cols": ["성과발생년도", "과제고유번호", "사업화형태", "사업화명", "기매출액(원)"],
        "label": "사업화",
    },
    "기술료": {
        "suffix": "기술료.xlsx",
        "cols": ["성과발생년도", "과제고유번호", "기술실시계약명", "당해연도 기술료(원)"],
        "label": "기술료",
    },
}


def _load_outcome_type(spec: dict) -> pd.DataFrame:
    """해당 유형의 전 연도 파일을 합쳐 DataFrame 반환. parquet 우선, xlsx 폴백."""
    frames = []
    suffix_parquet = spec["suffix"].replace(".xlsx", ".parquet")
    files = sorted(OUTCOME_DIR.glob(f"*{suffix_parquet}"))
    use_parquet = bool(files)
    if not files:
        files = sorted(OUTCOME_DIR.glob(f"*{spec['suffix']}"))

    for f in files:
        try:
            df = pd.read_parquet(f) if use_parquet else pd.read_excel(f)
            exist_cols = [c for c in spec["cols"] if c in df.columns]
            frames.append(df[exist_cols])
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


class ContinuationLookup:
    """애플리케이션 시작 시 한 번 로드, 이후 O(1) 조회."""

    _instance: "ContinuationLookup | None" = None

    @classmethod
    def get(cls) -> "ContinuationLookup":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._index: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        self._load()

    def _load(self):
        for key, spec in OUTCOME_SPECS.items():
            df = _load_outcome_type(spec)
            if df.empty or "과제고유번호" not in df.columns:
                continue
            for _, row in df.iterrows():
                pid = str(row.get("과제고유번호", "")).strip()
                if not pid or pid == "nan":
                    continue
                record = {c: row.get(c) for c in spec["cols"] if c in df.columns}
                self._index[pid][key].append(record)

    def lookup(self, project_id: str, before_year: int | None = None) -> str:
        """과제고유번호로 이전 성과를 조회해 요약 문자열 반환.

        before_year 가 주어지면 해당 연도보다 이전 성과만 반환.
        """
        pid = str(project_id).strip()
        outcomes = self._index.get(pid, {})
        if not outcomes:
            return ""

        lines = []
        for key, spec in OUTCOME_SPECS.items():
            records = outcomes.get(key, [])
            filtered = []
            for r in records:
                yr = r.get("성과발생년도")
                try:
                    if before_year and int(yr) >= before_year:
                        continue
                except (TypeError, ValueError):
                    pass
                filtered.append(r)
            if not filtered:
                continue
            lines.append(f"[{spec['label']}] {len(filtered)}건")
            for r in filtered[:3]:  # 최대 3건 상세
                yr = r.get("성과발생년도", "")
                if key == "SCI논문":
                    lines.append(
                        f"  · ({yr}) {r.get('논문명','')[:60]} "
                        f"— {r.get('학술지명','')}, SCI={r.get('SCI여부(입력시)','')}"
                    )
                elif key == "특허":
                    lines.append(
                        f"  · ({yr}) {r.get('발명의 명칭','')[:60]} "
                        f"[{r.get('출원/등록 구분','')} / {r.get('출원/등록 국가','')}]"
                    )
                elif key == "사업화":
                    sales = r.get("기매출액(원)", 0) or 0
                    lines.append(
                        f"  · ({yr}) {r.get('사업화명','')[:50]} "
                        f"({r.get('사업화형태','')}, 매출 {int(sales):,}원)"
                    )
                elif key == "기술료":
                    fee = r.get("당해연도 기술료(원)", 0) or 0
                    lines.append(
                        f"  · ({yr}) {r.get('기술실시계약명','')[:50]} "
                        f"(기술료 {int(fee):,}원)"
                    )
        return "\n".join(lines)
