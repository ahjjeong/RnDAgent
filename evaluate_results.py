"""Summarize prediction performance from a results.jsonl file.

Usage:
    python evaluate_results.py --file results.jsonl
    python evaluate_results.py --file results.jsonl --out metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate import summarize_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

    results = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = summarize_results(results)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)

    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Saved → {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
