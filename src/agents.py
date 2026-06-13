"""단계별 직종 기반 3인 평가단 + FacilitatorAgent(Coordinator·Moderator 겸임).

각 에이전트는 연구개발단계별로 배정된 전문 직종(panel_type) 관점에서
창의성·수행계획 충실성·연구개발 역량 전 항목을 평가합니다.
"""
from __future__ import annotations
from .llm import LocalLLM, extract_json
from .prompt_config import (
    AgentConfig, AGENT_MAP,
    CONTEXT_COLS, ITEM_COLS, EVAL_ITEMS, STAGE_NORMALIZE,
)
from .continuation_lookup import ContinuationLookup
from .web_rag import search_for_creativity


GRADE_OPTIONS = "매우 우수 | 우수 | 보통 | 미흡 | 매우 미흡"

_ITEM_BLOCK = (
    '    "grade": "' + GRADE_OPTIONS + '",\n'
    '    "reasoning": "2~3문장 (실제 수치·데이터 인용 필수)",\n'
    '    "evidence": {"데이터_항목": "실제_값"},\n'
    '    "strengths": ["강점"],\n'
    '    "weaknesses": ["약점"]'
)

OUTPUT_SCHEMA = (
    '{\n'
    '  "창의성": {\n' + _ITEM_BLOCK + '\n  },\n'
    '  "수행계획 충실성": {\n' + _ITEM_BLOCK + '\n  },\n'
    '  "연구개발 역량": {\n' + _ITEM_BLOCK + '\n  }\n'
    '}'
)


def _pick(row: dict, cols: list[str]) -> dict:
    return {c: v for c in cols if (v := row.get(c)) is not None}


def _fmt(fields: dict) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in fields.items()) or "(제공 정보 없음)"


def _normalize_stage(raw: str | None) -> str:
    if not raw:
        return "응용연구"
    for k, v in STAGE_NORMALIZE.items():
        if k in str(raw):
            return v
    return "응용연구"


# ─────────────────────────────────────────────────────────────────
class EvaluatorAgent:
    """슬롯 기반 평가 에이전트 — 전 평가 항목을 자신의 관점에서 평가."""

    def __init__(self, slot: str, llm: LocalLLM, retriever=None,
                 continuation_lookup: ContinuationLookup | None = None):
        self.slot = slot
        self.llm = llm
        self.retriever = retriever
        self.cont_lookup = continuation_lookup

    def _build_prompt(self, project: dict, cfg: AgentConfig, row: dict) -> tuple[str, str]:
        context = _pick(row, CONTEXT_COLS)
        stage   = cfg.stage
        is_cont = str(row.get("신규계속구분", "")).strip() == "계속"

        # ── 계속과제 성과 블록 (prefetch 우선) ──
        prior_block = ""
        if is_cont:
            cached = project.get("_rag_prior")
            if cached is not None:
                if cached == "__not_found__":
                    prior_block = (
                        "\n[계속과제 — 이전 성과 미확인]\n"
                        "※ 계속과제로 표시되었으나 데이터베이스에서 이전 성과를 찾지 못했습니다."
                    )
                else:
                    prior_block = (
                        "\n[계속과제 — 이전 연도 성과 기록]\n"
                        "※ 아래는 평가 대상 연도 이전에 확인된 성과입니다. "
                        "수행계획 충실성과 연구개발 역량 평가에 반영하십시오.\n" + cached
                    )
            elif self.cont_lookup is not None:
                pid = str(row.get("과제고유번호", ""))
                try:
                    before_year = int(row.get("과제수행연도", 0))
                except (TypeError, ValueError):
                    before_year = None
                prior = self.cont_lookup.lookup(pid, before_year=before_year)
                if prior:
                    prior_block = (
                        "\n[계속과제 — 이전 연도 성과 기록]\n"
                        "※ 아래는 평가 대상 연도 이전에 확인된 성과입니다. "
                        "수행계획 충실성과 연구개발 역량 평가에 반영하십시오.\n" + prior
                    )
                else:
                    prior_block = (
                        "\n[계속과제 — 이전 성과 미확인]\n"
                        "※ 계속과제로 표시되었으나 데이터베이스에서 이전 성과를 찾지 못했습니다."
                    )

        # ── 신규/계속 지침 ──
        if is_cont:
            continuity_guide = (
                "## 신규·계속 평가 지침\n"
                "이 과제는 계속과제입니다. 이전 성과가 제공된 경우:\n"
                "· 제안된 목표가 이전 성과의 연장선에 논리적으로 위치하는가\n"
                "· 이전 성과 수준이 목표 난이도와 부합하는가\n"
                "· 이전 성과가 부진하다면 현 제안의 실현 가능성을 보수적으로 판단하십시오"
            )
        else:
            continuity_guide = (
                "## 신규·계속 평가 지침\n"
                "이 과제는 신규과제입니다.\n"
                "· 과거 실적 대신 제안 자체의 논리적 완결성과 구체성을 중심으로 평가하십시오\n"
                "· 신규이기 때문에 불리하게 평가하는 것이 아니라, 제안 자체의 질로만 판단하십시오"
            )

        # ── 외부 RAG (prefetch 우선, 없으면 직접 호출) ──
        web_block = ""
        if "_rag_web" in project:
            web_result = project["_rag_web"]
            if web_result:
                web_block = "\n" + web_result
        else:
            keywords = " ".join(filter(None, [
                str(_pick(row, ITEM_COLS["창의성"]).get("요약문_한글키워드", "")),
                str(_pick(row, ITEM_COLS["창의성"]).get("요약문_연구목표", ""))[:100],
            ]))
            web_result = search_for_creativity(keywords, stage=stage, limit=3)
            if web_result:
                print(f"[RAG] 외부검색 {stage} — {len(web_result.splitlines())}줄 주입됨")
                web_block = "\n" + web_result

        # ── 내부 RAG (prefetch 우선, 없으면 직접 호출) ──
        ref_block = ""
        if "_rag_ref" in project:
            ref_block = project["_rag_ref"]
        elif self.retriever is not None:
            all_data_str = " ".join(
                str(v) for item in EVAL_ITEMS
                for v in _pick(row, ITEM_COLS[item]).values()
            )
            refs = self.retriever.search(self.slot, all_data_str, k=3)
            if refs:
                print(f"[RAG] 내부검색 {self.slot} — {len(refs)}건 주입됨")
                ref_block = "\n[유사 과거 과제 참고]\n" + "\n".join(f"* {r[:300]}" for r in refs)

        # ── 항목별 데이터 블록 ──
        creativity_data  = _fmt(_pick(row, ITEM_COLS["창의성"]))
        execution_data   = _fmt(_pick(row, ITEM_COLS["수행계획 충실성"]))
        capability_data  = _fmt(_pick(row, ITEM_COLS["연구개발 역량"]))
        cumulative_note  = (
            "※ 총연구기간·총연구비는 전체 과제 기간·비용의 누적값입니다.\n"
        )

        # ── System ──
        system = f"""{cfg.persona}

당신은 아래 세 평가 항목 전체를 {cfg.panel_type}(으)로서 독립적으로 평가합니다.
다른 위원의 관점을 대신 평가하거나 언급하지 마십시오."""

        # ── User ──
        user = f"""## 과제 맥락
- 연구개발단계: {context.get("연구개발단계(변경)", "(미상)")}
- 신규/계속 구분: {"계속과제" if is_cont else "신규과제"}
- 기술분야:
  · {context.get("과학기술표준분류1-대","")} > {context.get("과학기술표준분류1-중","")} > {context.get("과학기술표준분류1-소","")} (가중치 {context.get("과학기술표준분류가중치1","")}%)
  · {context.get("과학기술표준분류2-대","")} > {context.get("과학기술표준분류2-중","")} > {context.get("과학기술표준분류2-소","")} (가중치 {context.get("과학기술표준분류가중치2","")}%)
- 주관 부처: {context.get("부처명","")}  |  내역사업명: {context.get("내역사업명","")}

{continuity_guide}
{prior_block}

---

## 항목 1. 창의성 — 입력 데이터
{creativity_data}{web_block}{ref_block}

### 창의성 평가 초점 [{cfg.panel_type}]
{cfg.focus["창의성"]}

---

## 항목 2. 수행계획 충실성 — 입력 데이터
{cumulative_note}{execution_data}

### 수행계획 충실성 평가 초점 [{cfg.panel_type}]
{cfg.focus["수행계획 충실성"]}

---

## 항목 3. 연구개발 역량 — 입력 데이터
{capability_data}

### 연구개발 역량 평가 초점 [{cfg.panel_type}]
{cfg.focus["연구개발 역량"]}

---

## 평가 등급 기준 (전 항목 공통)
- 매우 우수: 동일 분야 상위권에 해당하는 탁월한 수준
- 우수: 기대 수준을 상회하며 명확한 강점이 확인됨
- 보통: 평균 수준으로 특별한 강점이나 약점이 두드러지지 않음
- 미흡: 일부 결함이 확인되며 보완이 필요
- 매우 미흡: 기대 수준에 크게 못 미치며 근본적 재검토가 필요

## 출력 형식 (JSON만 출력)
{OUTPUT_SCHEMA}"""

        return system, user

    def evaluate(self, project: dict) -> dict:
        row: dict = project.get("_raw_row", {})
        stage = _normalize_stage(row.get("연구개발단계(변경)"))
        cfg = AGENT_MAP.get((stage, self.slot)) or AGENT_MAP[("응용연구", self.slot)]

        system, user = self._build_prompt(project, cfg, row)
        raw = self.llm.chat(system, user, max_new_tokens=768)
        parsed = extract_json(raw)
        parsed["agent"]      = f"{stage}_{self.slot}_Agent"
        parsed["slot"]       = self.slot
        parsed["panel_type"] = cfg.panel_type
        return parsed


# ─────────────────────────────────────────────────────────────────
class FacilitatorAgent:
    """위원장 — Coordinator(쟁점 식별·질문) + Moderator(최종 판정) 겸임."""

    PERSONA = (
        "당신은 국가 R&D 과제 선정 위원회의 위원장입니다. "
        "공정성과 균형을 최우선으로 하며, 세 위원의 의견을 조율하고 최종 판정을 내립니다."
    )

    VERDICT_SCHEMA = (
        '{\n'
        '  "creativity_grade":  "매우 우수|우수|보통|미흡|매우 미흡",\n'
        '  "execution_grade":   "매우 우수|우수|보통|미흡|매우 미흡",\n'
        '  "capability_grade":  "매우 우수|우수|보통|미흡|매우 미흡",\n'
        '  "decision":          "선정|조건부 선정|비선정|판정 보류",\n'
        '  "deferral_reason":   "판정 보류 사유 (해당 없으면 null)",\n'
        '  "conditions":        ["조건 (없으면 빈 배열)"],\n'
        '  "priority_rank":     "A|B|C|D",\n'
        '  "key_debate_points": "핵심 쟁점 2~3문장",\n'
        '  "consensus_level":   "full|partial|none",\n'
        '  "confidence":        0.0\n'
        '}'
    )

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    # ── Coordinator 역할 ──
    def identify_issues(self, evaluations: list[dict]) -> str:
        # 항목별로 각 위원 등급·근거 한 줄 요약 구성
        per_item_lines = []
        for item in EVAL_ITEMS:
            per_item_lines.append(f"[{item}]")
            for e in evaluations:
                label      = e.get("panel_type") or e.get("slot", "?")
                item_eval  = e.get(item, {})
                grade      = item_eval.get("grade", "?")
                reasoning  = (item_eval.get("reasoning") or "")[:80]
                per_item_lines.append(f"{label}({grade}): {reasoning}")
            per_item_lines.append("")

        # 위원 직함 목록 (동적)
        labels = [e.get("panel_type") or e.get("slot", "위원") for e in evaluations]
        label_fmt = "\n".join(f"{l}(등급): 핵심 판단 근거 한 줄" for l in labels)

        system = (
            f"{self.PERSONA}\n\n"
            "1차 평가 결과를 항목별로 정리했습니다. 아래 형식으로 출력하십시오:\n\n"
            "[항목명]\n"
            f"{label_fmt}\n"
            "쟁점: 의견 차이가 가장 두드러지는 포인트를 한 줄 질문으로\n\n"
            "세 항목 모두 정리한 후, 각 위원에게 확인 요청 질문을 한 줄씩 제시하십시오."
        )
        user = "1차 평가 결과:\n\n" + "\n".join(per_item_lines)
        return self.llm.chat(system, user, max_new_tokens=800)

    # ── Moderator 역할 ──
    def decide(self, project: dict, evaluations: list[dict],
               coordinator_issues: str, rebuttals: list[dict]) -> dict:
        eval_summary = "\n".join(
            f"{e.get('panel_type', e.get('slot','?'))}:\n" + "\n".join(
                f"  {item}: {e.get(item, {}).get('grade','?')} — "
                f"{(e.get(item, {}).get('reasoning') or '')[:80]}…"
                for item in EVAL_ITEMS
            )
            for e in evaluations
        )
        rebuttal_summary = "\n".join(
            f"- {r.get('panel_type', r.get('slot','?'))}: {r.get('response','')}"
            for r in rebuttals
        )
        system = (
            f"{self.PERSONA}\n"
            "Round 2 토론까지 완료되었습니다. "
            "신뢰도(confidence)가 0.5 미만이거나 핵심 정보가 불충분하면 "
            "decision을 '판정 보류'로 설정하고 deferral_reason에 사유를 기재하십시오. "
            "그 외에는 아래 JSON 스키마로만 최종 판정하세요."
        )
        user = (
            f"[과제명] {project.get('title')}\n\n"
            f"[1차 평가 요약]\n{eval_summary}\n\n"
            f"[Coordinator 쟁점]\n{coordinator_issues}\n\n"
            f"[Round 2 반론]\n{rebuttal_summary}\n\n"
            f"[출력 스키마 — JSON만 응답]\n{self.VERDICT_SCHEMA}"
        )
        raw = self.llm.chat(system, user, max_new_tokens=700)
        return extract_json(raw)


# ─────────────────────────────────────────────────────────────────
class DebateAgent:
    """Round 2 — 슬롯 기반 반론 에이전트."""

    def __init__(self, slot: str, llm: LocalLLM):
        self.slot = slot
        self.llm = llm

    def respond(self, project: dict, issues: str, peer_evals: list[dict]) -> dict:
        row: dict = project.get("_raw_row", {})
        stage = _normalize_stage(row.get("연구개발단계(변경)"))
        cfg = AGENT_MAP.get((stage, self.slot)) or AGENT_MAP[("응용연구", self.slot)]

        # 동료 평가 요약 (항목별)
        peer_lines = []
        for e in peer_evals:
            label = e.get("panel_type") or e.get("slot", "?")
            peer_lines.append(f"[{label}]")
            for item in EVAL_ITEMS:
                item_eval = e.get(item, {})
                peer_lines.append(
                    f"  {item}: {item_eval.get('grade','?')} — "
                    f"{(item_eval.get('reasoning') or '')[:100]}"
                )
        peer_text = "\n".join(peer_lines)

        system = (
            f"{cfg.persona}\n"
            f"당신은 {cfg.panel_type} 위원입니다. "
            f"Coordinator의 질문 중 당신에게 향한 부분에 답하고, "
            f"필요하다면 다른 위원의 평가에 대해 {cfg.panel_type} 관점에서만 보완 의견을 제시하십시오."
        )
        user = (
            f"[과제명] {project.get('title')}\n\n"
            f"[동료 평가 요약]\n{peer_text}\n\n"
            f"[Coordinator 쟁점 및 질문]\n{issues}\n\n"
            f"위 내용에 대해 {cfg.panel_type} 관점에서 3문장 이내로 답변하십시오. "
            f"JSON 아닌 일반 텍스트."
        )
        resp = self.llm.chat(system, user, max_new_tokens=512)
        return {
            "agent":      f"{stage}_{self.slot}_Agent",
            "slot":       self.slot,
            "panel_type": cfg.panel_type,
            "response":   resp,
        }
