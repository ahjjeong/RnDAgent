"""LangGraph 기반 3-Phase 선정 파이프라인.

그래프 토폴로지:

        START
          |
       prefetch  (RAG 1회 실행 → project에 캐시)
       /  |  \
      / phase1 \
     / (3병렬) \
     \   |   /
      → coordinator
     /   |   \
    / phase2 \
    \ (3병렬) /
      → moderator
          |
         END
"""
from __future__ import annotations
import operator
from typing import Annotated, Any, TypedDict
from langgraph.graph import StateGraph, START, END

from .llm import LocalLLM
from .agents import EvaluatorAgent, FacilitatorAgent, DebateAgent
from .prompt_config import AGENT_SLOTS, ITEM_COLS, STAGE_NORMALIZE
from .continuation_lookup import ContinuationLookup
from .web_rag import search_for_creativity


def _normalize_stage(raw: str | None) -> str:
    if not raw:
        return "응용연구"
    for k, v in STAGE_NORMALIZE.items():
        if k in str(raw):
            return v
    return "응용연구"


class SelectionState(TypedDict, total=False):
    project: dict[str, Any]
    phase1: Annotated[list[dict], operator.add]
    coordinator_issues: str
    phase2_rebuttals: Annotated[list[dict], operator.add]
    verdict: dict


EventCallback = Any


def build_graph(retriever=None, event_cb: EventCallback = None):
    llm = LocalLLM.get()
    cont_lookup = ContinuationLookup.get()

    evaluators = {
        slot: EvaluatorAgent(slot, llm, retriever, continuation_lookup=cont_lookup)
        for slot in AGENT_SLOTS
    }
    debaters = {slot: DebateAgent(slot, llm) for slot in AGENT_SLOTS}
    facilitator = FacilitatorAgent(llm)

    def emit(event: str, data: dict):
        if event_cb:
            event_cb(event, data)

    g = StateGraph(SelectionState)

    # ---- Prefetch: RAG 1회 실행 (모든 에이전트가 공유) ----
    def prefetch_node(state: SelectionState) -> dict:
        project = state["project"]
        row: dict = project.get("_raw_row", {})
        stage = _normalize_stage(row.get("연구개발단계(변경)"))

        # 외부 RAG (arXiv / KIPRIS) — 캐시 포함 1회만
        creativity_row = {c: row.get(c) for c in ITEM_COLS["창의성"] if row.get(c)}
        keywords = " ".join(filter(None, [
            str(creativity_row.get("요약문_한글키워드", "")),
            str(creativity_row.get("요약문_연구목표", ""))[:100],
        ]))
        web_result = search_for_creativity(keywords, stage=stage, limit=3)
        if web_result:
            print(f"[RAG] 외부검색 prefetch {stage} — {len(web_result.splitlines())}줄")

        # 내부 RAG — 에이전트별로 쿼리가 동일하므로 1회 계산
        all_data_str = " ".join(
            str(v) for item in ["창의성", "수행계획 충실성", "연구개발 역량"]
            for v in {c: row.get(c) for c in ITEM_COLS[item] if row.get(c)}.values()
        )
        ref_result = ""
        if retriever is not None:
            refs = retriever.search("공통", all_data_str, k=3)
            if refs:
                print(f"[RAG] 내부검색 prefetch — {len(refs)}건")
                ref_result = "\n[유사 과거 과제 참고]\n" + "\n".join(f"* {r[:300]}" for r in refs)

        # 계속과제 성과 블록
        is_cont = str(row.get("신규계속구분", "")).strip() == "계속"
        prior_result = ""
        if is_cont and cont_lookup is not None:
            pid = str(row.get("과제고유번호", ""))
            try:
                before_year = int(row.get("과제수행연도", 0))
            except (TypeError, ValueError):
                before_year = None
            prior = cont_lookup.lookup(pid, before_year=before_year)
            prior_result = prior if prior else "__not_found__"

        updated = {
            **project,
            "_rag_web": web_result,
            "_rag_ref": ref_result,
            "_rag_prior": prior_result,
        }
        return {"project": updated}

    g.add_node("prefetch", prefetch_node)
    g.add_edge(START, "prefetch")

    # ---- Phase 1: 병렬 독립 평가 ----
    def make_phase1_node(slot: str):
        def _node(state: SelectionState) -> dict:
            emit("phase1_start", {"slot": slot})
            result = evaluators[slot].evaluate(state["project"])
            emit("phase1_done", {"slot": slot, "result": result})
            return {"phase1": [result]}
        return _node

    for slot in AGENT_SLOTS:
        node_id = f"phase1__{slot}"
        g.add_node(node_id, make_phase1_node(slot))
        g.add_edge("prefetch", node_id)

    # ---- Coordinator ----
    def coordinator_node(state: SelectionState) -> dict:
        emit("coordinator_start", {})
        issues = facilitator.identify_issues(state["phase1"])
        emit("coordinator_done", {"issues": issues})
        return {"coordinator_issues": issues}

    g.add_node("coordinator", coordinator_node)
    for slot in AGENT_SLOTS:
        g.add_edge(f"phase1__{slot}", "coordinator")

    # ---- Phase 2: 병렬 반론 ----
    def make_phase2_node(slot: str):
        def _node(state: SelectionState) -> dict:
            emit("phase2_start", {"slot": slot})
            resp = debaters[slot].respond(
                state["project"], state["coordinator_issues"], state["phase1"]
            )
            emit("phase2_done", {"slot": slot, "result": resp})
            return {"phase2_rebuttals": [resp]}
        return _node

    for slot in AGENT_SLOTS:
        node_id = f"phase2__{slot}"
        g.add_node(node_id, make_phase2_node(slot))
        g.add_edge("coordinator", node_id)

    # ---- Moderator ----
    def moderator_node(state: SelectionState) -> dict:
        emit("moderator_start", {})
        verdict = facilitator.decide(
            state["project"],
            state["phase1"],
            state["coordinator_issues"],
            state["phase2_rebuttals"],
        )
        emit("moderator_done", {"verdict": verdict})
        return {"verdict": verdict}

    g.add_node("moderator", moderator_node)
    for slot in AGENT_SLOTS:
        g.add_edge(f"phase2__{slot}", "moderator")
    g.add_edge("moderator", END)

    return g.compile()


class SelectionPipeline:
    def __init__(self, retriever=None, event_cb: EventCallback = None):
        self.app = build_graph(retriever=retriever, event_cb=event_cb)

    def run(self, project: dict) -> dict:
        final = self.app.invoke({"project": project})
        return {
            "project_id":         project.get("id"),
            "title":              project.get("title"),
            "phase1":             final.get("phase1", []),
            "coordinator_issues": final.get("coordinator_issues", ""),
            "phase2_rebuttals":   final.get("phase2_rebuttals", []),
            "verdict":            final.get("verdict", {}),
        }
