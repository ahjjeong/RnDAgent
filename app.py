"""FastAPI demo server — real-time R&D selection pipeline with SSE streaming."""
from __future__ import annotations
import json
import queue
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from src.data_loader import load_all, project_views

app = FastAPI(title="R&D 선정 에이전트 데모")

_projects: list[dict] = []


@app.on_event("startup")
def _startup():
    global _projects

    # 1) 데이터셋 로드
    print("[app] 데이터셋 로딩 중…")
    try:
        df = load_all(years=[2023])
    except Exception:
        df = load_all()
    n = min(5, len(df))
    sample = df.sample(n=n, random_state=42).reset_index(drop=True)
    _projects = [project_views(row) for _, row in sample.iterrows()]
    print(f"[app] 과제 {len(_projects)}개 로드 완료")

    # 2) LLM 싱글톤 preload (첫 요청 지연 방지)
    print("[app] LLM 모델 로딩 중…")
    from src.llm import LocalLLM
    LocalLLM.get()
    print("[app] LLM 로딩 완료")

    # 3) ContinuationLookup preload
    print("[app] 성과 데이터 로딩 중…")
    from src.continuation_lookup import ContinuationLookup
    ContinuationLookup.get()
    print("[app] 성과 데이터 로딩 완료")


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/projects")
def get_projects():
    return [
        {"idx": i, "id": str(p.get("id", "")), "title": str(p.get("title", f"과제 {i+1}"))}
        for i, p in enumerate(_projects)
    ]


_runs: dict[str, queue.Queue] = {}


@app.post("/run/{idx}")
def start_run(idx: int):
    if idx < 0 or idx >= len(_projects):
        return {"error": "잘못된 과제 인덱스"}
    run_id = uuid.uuid4().hex[:8]
    q: queue.Queue = queue.Queue()
    _runs[run_id] = q

    project = _projects[idx]

    def _worker():
        # Import here to avoid loading models at startup
        from src.graph import SelectionPipeline

        def cb(event: str, data: dict):
            q.put_nowait({"event": event, "data": data})

        try:
            pipe = SelectionPipeline(event_cb=cb)
            result = pipe.run(project)
            q.put_nowait({"event": "done", "data": result})
        except Exception as exc:
            q.put_nowait({"event": "error", "data": {"message": str(exc)}})
        finally:
            q.put_nowait(None)  # sentinel

    threading.Thread(target=_worker, daemon=True).start()
    return {"run_id": run_id}


@app.get("/stream/{run_id}")
def stream_events(run_id: str):
    q = _runs.get(run_id)

    if q is None:
        def _err():
            yield 'data: {"event":"error","data":{"message":"run_id를 찾을 수 없습니다"}}\n\n'
        return StreamingResponse(_err(), media_type="text/event-stream")

    def _gen():
        while True:
            try:
                item = q.get(timeout=180)
            except queue.Empty:
                yield 'data: {"event":"keepalive"}\n\n'
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        _runs.pop(run_id, None)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
