"""xlsx → Parquet 일괄 변환 스크립트. 최초 1회만 실행하면 됩니다.

실행:
    python convert_to_parquet.py

변환 대상:
  dataset/*.xlsx          → dataset/*.parquet   (컬럼 정규화 포함)
  dataset/성과/*.xlsx     → dataset/성과/*.parquet
"""
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OUTCOME_DIR = DATASET_DIR / "성과"


def _read_data_sheet(path: Path) -> pd.DataFrame:
    """시트가 2개 이상이면 마지막 시트(실데이터)만 읽는다."""
    xf = pd.ExcelFile(path)
    sheet = xf.sheet_names[-1] if len(xf.sheet_names) > 1 else xf.sheet_names[0]
    return pd.read_excel(xf, sheet_name=sheet)


def convert_main():
    files = sorted(DATASET_DIR.glob("*.xlsx"))
    if not files:
        print("[main] xlsx 파일 없음")
        return

    from src.data_loader import _normalize_columns

    for f in files:
        out = f.with_suffix(".parquet")
        if out.exists():
            print(f"[main] 스킵 (이미 존재): {out.name}")
            continue
        print(f"[main] 변환 중: {f.name}  ({f.stat().st_size // 1_000_000} MB)", end="", flush=True)
        t0 = time.time()
        df = _read_data_sheet(f)
        df = _normalize_columns(df)
        df.to_parquet(out, index=False)
        print(f"  →  {out.name}  ({out.stat().st_size // 1_000_000} MB)  {time.time()-t0:.1f}s")


def convert_outcomes():
    files = sorted(OUTCOME_DIR.glob("*.xlsx"))
    if not files:
        print("[성과] xlsx 파일 없음")
        return

    for f in files:
        out = f.with_suffix(".parquet")
        if out.exists():
            print(f"[성과] 스킵: {out.name}")
            continue
        print(f"[성과] 변환 중: {f.name}", end="", flush=True)
        t0 = time.time()
        df = _read_data_sheet(f)
        df.to_parquet(out, index=False)
        print(f"  →  {out.name}  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    print("=== xlsx → Parquet 변환 시작 ===\n")
    t_all = time.time()
    convert_main()
    print()
    convert_outcomes()
    print(f"\n=== 완료  총 {time.time()-t_all:.1f}s ===")
