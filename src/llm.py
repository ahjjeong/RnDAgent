"""LLM wrapper supporting vLLM OpenAI-compatible API and HF transformers."""
from __future__ import annotations
import json
import re
import threading
from urllib import error, request

from .config import (
    LLM_BACKEND,
    LLM_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_TIMEOUT_SECONDS,
    VLLM_ENABLE_THINKING,
    VLLM_REASONING_EFFORT,
    VLLM_THINKING_TOKEN_BUDGET,
)


class LocalLLM:
    _instance = None

    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.backend = LLM_BACKEND.lower().strip()
        self._lock = threading.Lock()
        if self.backend == "transformers":
            self._init_transformers(model_name)
        elif self.backend != "vllm":
            raise ValueError(f"Unsupported LLM_BACKEND={LLM_BACKEND!r}")

    def _init_transformers(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    @classmethod
    def get(cls) -> "LocalLLM":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def chat(
        self,
        system: str,
        user: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        json_mode: bool = False,
    ) -> str:
        if self.backend == "vllm":
            return self._chat_vllm(system, user, max_new_tokens, json_mode=json_mode)
        return self._chat_transformers(system, user, max_new_tokens)

    def _chat_vllm(
        self,
        system: str,
        user: str,
        max_new_tokens: int,
        json_mode: bool = False,
    ) -> str:
        if json_mode:
            system = (
                system
                + "\n\n응답 규칙: 사고 과정, 해설, Markdown 없이 유효한 JSON 객체 하나만 출력하십시오."
            )
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_new_tokens,
            "temperature": TEMPERATURE,
            "top_p": 0.9,
        }
        if VLLM_REASONING_EFFORT:
            payload["reasoning_effort"] = VLLM_REASONING_EFFORT
        if VLLM_THINKING_TOKEN_BUDGET:
            payload["thinking_token_budget"] = int(VLLM_THINKING_TOKEN_BUDGET)
        if VLLM_ENABLE_THINKING:
            payload["chat_template_kwargs"] = {
                "enable_thinking": VLLM_ENABLE_THINKING.lower() in {"1", "true", "yes", "y"}
            }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{VLLM_BASE_URL.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VLLM_API_KEY}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=VLLM_TIMEOUT_SECONDS) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(
                f"vLLM 서버 HTTP {exc.code} 오류: {VLLM_BASE_URL}. {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"vLLM 서버 호출 실패: {VLLM_BASE_URL}. "
                "먼저 GPU 0-3에서 vLLM 서버를 띄웠는지 확인하세요."
            ) from exc
        return data["choices"][0]["message"]["content"].strip()

    def _chat_transformers(self, system: str, user: str, max_new_tokens: int) -> str:
        import torch

        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        with self._lock, torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=TEMPERATURE > 0,
                temperature=max(TEMPERATURE, 1e-5),
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            text = self.tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
        return text.strip()


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"_raw": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([\]}])", r"\1", m.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"_raw": text}
