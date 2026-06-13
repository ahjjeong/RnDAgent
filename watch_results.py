"""results.jsonl 을 실시간으로 tail 하며 예쁘게 출력.

사용법:
    # 터미널 1: 파이프라인 실행
    python main.py --n 10 --year 2023 --out results.jsonl

    # 터미널 2: 실시간 관전
    python watch_results.py
    python watch_results.py --file results.jsonl --phase 3
"""
import argparse
import json
import time
from pathlib import Path
from view_results import show


def tail(path: Path, phase: int | None, raw: bool, from_start: bool):
    print(f"[watch] {path}  (Ctrl-C 로 종료)")
    # 파일이 아직 없을 수도 있음 — 생길 때까지 대기
    while not path.exists():
        time.sleep(0.5)

    with path.open("r", encoding="utf-8") as f:
        if not from_start:
            f.seek(0, 2)  # 끝으로 이동
        idx = 0
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            show(r, idx, phase, raw)
            idx += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results.jsonl")
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], default=None)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--from-start", action="store_true",
                    help="처음부터 다시 출력 (기본은 새로 추가되는 줄만)")
    args = ap.parse_args()
    try:
        tail(Path(args.file), args.phase, args.raw, args.from_start)
    except KeyboardInterrupt:
        print("\n[watch] stopped")


if __name__ == "__main__":
    main()
