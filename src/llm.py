"""Lightweight local LLM wrapper around HF transformers."""
from __future__ import annotations
import json
import re
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE


class LocalLLM:
    _instance = None

    def __init__(self, model_name: str = LLM_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "LocalLLM":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @torch.inference_mode()
    def chat(self, system: str, user: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        with self._lock:
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
