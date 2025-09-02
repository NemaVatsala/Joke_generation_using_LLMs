from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def distinct_n(tokens: Sequence[str], n: int) -> float:
    """Compute distinct-n (ratio of unique n-grams). Encourages diversity."""
    if n <= 0:
        return 0.0
    if len(tokens) < n:
        return 0.0
    ngrams = {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
    return len(ngrams) / max(1, len(tokens) - n + 1)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def clip(s: str, n: int = 240) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# -------------------------
# LLM client abstraction
# -------------------------

class LLMClient:
    """Provider-agnostic async chat + embeddings interface.

    Why: Keeps the pipeline decoupled from any vendor and supports
    novelty checks via a common `aembed()` API.
    """

    async def acomplete(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

    async def aembed(self, texts: List[str], model: str) -> List[List[float]]:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI adapter (optional)."""

    def __init__(self):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pip install openai; set OPENAI_API_KEY") from e
        self._client = OpenAI()

    async def acomplete(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        import asyncio

        def _run_sync() -> str:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}] + messages,
                temperature=temperature,
                top_p=top_p,
                **(extra or {}),
            )
            return resp.choices[0].message.content or ""

        return await asyncio.to_thread(_run_sync)

    async def aembed(self, texts: List[str], model: str) -> List[List[float]]:
        import asyncio

        def _run_sync() -> List[List[float]]:
            resp = self._client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]

        return await asyncio.to_thread(_run_sync)


class HuggingFaceClient(LLMClient):
    """Hugging Face transformers adapter for open-source models.

    Requirements: `pip install transformers torch accelerate`
    - Chat: uses `text-generation` with chat templates when available.
    - Embeddings: loads any encoder or causal LM and returns mean-pooled token embeddings.
    """

    def __init__(self) -> None:
        self._gen_pipes: Dict[str, Any] = {}
        self._gen_toks: Dict[str, Any] = {}
        self._emb_models: Dict[str, Any] = {}
        self._emb_toks: Dict[str, Any] = {}

    # ---- Chat ----
    def _get_gen(self, model: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        if model in self._gen_pipes:
            return self._gen_pipes[model], self._gen_toks[model]
        dtype = torch.bfloat16 if torch.cuda.is_available() else None
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype=dtype, trust_remote_code=True
        )
        pipe = pipeline(
            task="text-generation",
            model=mdl,
            tokenizer=tok,
            device_map="auto",
            return_full_text=True,
        )
        self._gen_pipes[model] = pipe
        self._gen_toks[model] = tok
        return pipe, tok

    @staticmethod
    def _chat_prompt(system: str, messages: List[Dict[str, str]], tokenizer: Any) -> str:
        convo: List[Dict[str, str]] = []
        if system:
            convo.append({"role": "system", "content": system})
        convo.extend({"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages)
        try:
            return tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        except Exception:
            parts: List[str] = []
            if system:
                parts.append(f"[SYSTEM]{system}[/SYSTEM]")
            for m in messages:
                r, c = m.get("role", "user"), m.get("content", "")
                parts.append(f"{r.title()}: {c}")
            parts.append("Assistant:")
            return "".join(parts)

    async def acomplete(
        self,
        *,
        system: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        import asyncio
        pipe, tok = self._get_gen(model)
        prompt = self._chat_prompt(system, messages, tok)
        max_new_tokens = int((extra or {}).get("max_tokens", 256))

        def _run_sync() -> str:
            out = pipe(
                prompt,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=max_new_tokens,
                pad_token_id=getattr(tok, "eos_token_id", None),
                eos_token_id=getattr(tok, "eos_token_id", None),
            )[0]["generated_text"]
            return out[len(prompt):].strip() if isinstance(out, str) and out.startswith(prompt) else str(out).strip()

        return await asyncio.to_thread(_run_sync)

    # ---- Embeddings ----
    def _get_embed(self, model: str):
        from transformers import AutoModel, AutoTokenizer
        import torch
        if model in self._emb_models:
            return self._emb_models[model], self._emb_toks[model]
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(model, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)
        mdl.eval()
        self._emb_models[model] = (mdl, device)
        self._emb_toks[model] = tok
        return (mdl, device), tok

    async def aembed(self, texts: List[str], model: str) -> List[List[float]]:
        import asyncio, torch
        (mdl, device), tok = self._get_embed(model)

        def _run_sync() -> List[List[float]]:
            with torch.no_grad():
                enc = tok(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(device)
                out = mdl(**enc)
                last = out.last_hidden_state  # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)
                summed = (last * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-6)
                mean = (summed / denom).detach().cpu().tolist()
                return mean
        return await asyncio.to_thread(_run_sync)

