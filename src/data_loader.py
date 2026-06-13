"""Load R&D project xlsx files and expose per-agent views."""
from __future__ import annotations
import pandas as pd
from typing import Iterable
from .config import (
    DATASET_DIR, SHEET_NAME, AGENT1_COLS, AGENT2_COLS, AGENT3_COLS,
    ID_COL, TITLE_COL, COLUMN_ALIASES,
)

# 역방향 조회: 구 컬럼명 → 정규 컬럼명
_OLD_TO_CANONICAL: dict[str, str] = {
    old: canonical
    for canonical, olds in COLUMN_ALIASES.items()
    for old in olds
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """구 연도 컬럼명을 정규 명칭으로 rename. 이미 정규 명칭이 있으면 구 명칭 컬럼은 삭제."""
    rename_map = {}
    for old_col, canonical in _OLD_TO_CANONICAL.items():
        if old_col in df.columns:
            if canonical not in df.columns:
                rename_map[old_col] = canonical   # rename
            else:
                df = df.drop(columns=[old_col])   # 정규 명칭 이미 존재 → 구 컬럼 제거
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_all(years: Iterable[int] | None = None) -> pd.DataFrame:
    # parquet 우선, 없으면 xlsx 폴백
    parquet_files = sorted(DATASET_DIR.glob("*.parquet"))
    xlsx_files    = sorted(DATASET_DIR.glob("*.xlsx"))
    use_parquet   = bool(parquet_files)
    files = parquet_files if use_parquet else xlsx_files

    frames = []
    for f in files:
        if years and not any(str(y) in f.name for y in years):
            continue
        if use_parquet:
            df = pd.read_parquet(f)
        else:
            df = pd.read_excel(f, sheet_name=SHEET_NAME)
            df = _normalize_columns(df)
        df["__source_file"] = f.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No dataset files found in {DATASET_DIR}")
    return pd.concat(frames, ignore_index=True)


def _pick(row: pd.Series, cols: list[str]) -> dict:
    out = {}
    for c in cols:
        if c in row.index:
            v = row[c]
            if pd.isna(v):
                continue
            out[c] = v
    return out


def project_views(row: pd.Series) -> dict:
    """Return per-agent slices + identifier for a single project row."""
    return {
        "id": row.get(ID_COL),
        "title": row.get(TITLE_COL),
        "agent1": _pick(row, AGENT1_COLS),
        "agent2": _pick(row, AGENT2_COLS),
        "agent3": _pick(row, AGENT3_COLS),
        "_raw_row": row.to_dict(),  # 동적 페르소나 선택에 필요한 전체 row
    }


def format_fields(fields: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in fields.items())
