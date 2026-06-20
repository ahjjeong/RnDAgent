"""LangGraph 기반 3-Phase 성과예측 파이프라인.

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

from .llm import LocalLLM, extract_json
from .agents import EvaluatorAgent, FacilitatorAgent, DebateAgent
from .prompt_config import AGENT_SLOTS, ITEM_COLS, STAGE_NORMALIZE, TARGET_RESEARCH_STAGE
from .config import ID_COL, WEB_RAG_ENABLED, WEB_RAG_GATE_MAX_TOKENS, WEB_RAG_ON_DEMAND
from .continuation_lookup import ContinuationLookup
from .web_rag import search_for_creativity


def _normalize_stage(raw: str | None) -> str:
    if not raw:
        return TARGET_RESEARCH_STAGE
    for k, v in STAGE_NORMALIZE.items():
        if k in str(raw):
            return v
    return TARGET_RESEARCH_STAGE


def _project_year(row: dict) -> int | None:
    for col in ("종료연도", "project_end_year", "총연구기간-종료연월일", "과제수행연도", "제출년도"):
        value = row.get(col)
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            text = str(value).strip()
            if len(text) >= 4 and text[:4].isdigit():
                return int(text[:4])
    return None


def _should_call_external_rag(
    llm: LocalLLM,
    row: dict,
    keywords: str,
    refs: list[str],
    ref_block: str,
) -> tuple[bool, str]:
    """Ask the LLM whether external RAG is needed after internal prior-project RAG."""
    title = row.get("연구개발과제명-국문") or row.get("과제명") or ""
    year = _project_year(row)
    field = " > ".join(
        str(row.get(c, "") or "")
        for c in ("과학기술표준분류1-대", "과학기술표준분류1-중", "과학기술표준분류1-소")
    )
    system = (
        "당신은 생명과학 기초연구 과제의 창의성 평가를 위한 RAG 사용 여부를 결정하는 보조 평가자입니다. "
        "내부 RAG 결과는 성과 누출 방지를 위해 현재 과제 종료연도보다 이전 과제만 포함합니다. "
        "외부 RAG는 비용이 크므로, 내부 근거만으로 선행연구 대비 차별성 판단이 어렵다고 볼 때만 요청하십시오. "
        "중요: 내부 유사 과제가 적거나 없다는 사실만으로 외부 RAG를 요청하지 마십시오. "
        "그 경우는 오히려 참신성의 신호일 수 있습니다."
    )
    user = f"""아래 과제의 창의성 평가에 외부 논문 검색(arXiv)이 필요한지 판단하십시오.

[과제]
- 제목: {title}
- 연도: {year}
- 분야: {field}
- 키워드/목표 요약: {keywords}

[내부 RAG 결과]
{ref_block or "(내부 유사 과거 과제 없음)"}

판단 기준:
- 내부 RAG와 과제 설명만으로 차별성/창의성 판단이 가능하면 false
- 연구 키워드가 너무 일반적이거나, 내부 과제와의 비교만으로 선행논문 대비 novelty 판단이 위험하면 true
- 내부 결과가 적다는 이유만으로 true를 선택하지 말 것
- 외부 검색은 창의성 판단의 불확실성을 실질적으로 줄일 때만 true

JSON만 출력:
{{"use_external_rag": true, "reason": "1문장"}}"""
    raw = llm.chat(
        system,
        user,
        max_new_tokens=WEB_RAG_GATE_MAX_TOKENS,
        json_mode=True,
    )
    parsed = extract_json(raw)
    decision = parsed.get("use_external_rag", False)
    if isinstance(decision, str):
        decision = decision.strip().lower() in {"true", "1", "yes", "y"}
    return bool(decision), str(parsed.get("reason", ""))


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

        # RAG query text
        creativity_row = {c: row.get(c) for c in ITEM_COLS["창의성"] if row.get(c)}
        keywords = " ".join(filter(None, [
            str(creativity_row.get("요약문_한글키워드", "")),
            str(creativity_row.get("요약문_연구목표", ""))[:100],
        ]))

        # 내부 RAG — 실제 성과를 붙이므로 현재 과제 종료연도 이전만 검색해 성과 누출 방지
        all_data_str = " ".join(
            str(v) for item in ["창의성", "수행계획 충실성", "연구개발 역량"]
            for v in {c: row.get(c) for c in ITEM_COLS[item] if row.get(c)}.values()
        )
        ref_result = ""
        refs: list[str] = []
        if retriever is not None:
            refs = retriever.search(
                "공통",
                all_data_str,
                k=3,
                exclude_idx=row.get("__dataset_row_id"),
                max_year=(_project_year(row) - 1) if _project_year(row) is not None else None,
                exclude_project_id=row.get(ID_COL),
            )
            if refs:
                print(f"[RAG] 내부검색 prefetch — {len(refs)}건")
                ref_result = "\n[유사 과거 과제 참고: 실제 논문 성과 포함]\n" + "\n".join(f"* {r[:900]}" for r in refs)

        # 외부 RAG — 항상 호출하지 않고, LLM이 내부 RAG만으로 부족하다고 판단할 때 보강
        rag_gate_reason = ""
        if WEB_RAG_ENABLED:
            should_call_web = True
            rag_gate_reason = "WEB_RAG_ENABLED"
        elif retriever is not None and WEB_RAG_ON_DEMAND:
            should_call_web, rag_gate_reason = _should_call_external_rag(
                llm, row, keywords, refs, ref_result
            )
        else:
            should_call_web = False
        web_result = search_for_creativity(keywords, stage=stage, limit=3) if should_call_web else ""
        if web_result:
            reason = "on-demand" if not WEB_RAG_ENABLED else "enabled"
            print(
                f"[RAG] 외부검색 prefetch {stage} ({reason}) — "
                f"{len(web_result.splitlines())}줄 / 판단: {rag_gate_reason}"
            )

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
            "_rag_web_decision": {
                "called": bool(web_result),
                "reason": rag_gate_reason,
            },
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
