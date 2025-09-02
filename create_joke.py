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
from utils import * 
from joke_plan import *
from ranker import * 


class JokePipeline:
    """End-to-end runner that ties generation, judging, and ranking.

    Why: Keeps the CLI small and enables programmatic usage.
    """

    def __init__(self, llm: LLMClient, gen_cfg: GenConfig, judge_cfg: JudgeConfig, run_cfg: RunConfig):
        self.llm = llm
        self.gen = PlanSearch(llm, gen_cfg)
        self.judge = BiasAwareJudge(llm, judge_cfg)
        self.rank = EloRanker(k=judge_cfg.k_factor)
        self.run = run_cfg
        self.log_dir = run_cfg.outdir / f"run_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def _schedule_pairs(self, cands: List[JokeCandidate], budget: int) -> List[Tuple[int, int]]:
        """Create a diversified pair schedule with light Swiss-style sampling.

        Why: Spreads comparisons while focusing more matches near the current
        skill estimate, improving ranking efficiency under limited budgets.
        """
        n = len(cands)
        idx = list(range(n))
        pairs: set[Tuple[int, int]] = set()
        # seed a round-robin subset
        for i in range(n):
            for j in range(i + 1, min(n, i + 1 + n // 4)):
                if len(pairs) >= budget:
                    break
                pairs.add((i, j))
            if len(pairs) >= budget:
                break
        # fill randomly
        while len(pairs) < budget:
            a, b = random.sample(idx, 2)
            a, b = min(a, b), max(a, b)
            pairs.add((a, b))
        return list(pairs)

    async def run_topic(self, topic: str) -> Dict[str, Any]:
        set_seed(self.run.seed)
        meta = {"topic": topic, "seed": self.run.seed, "started": now_iso()}
        # 1) Generate candidates
        cands = await self.gen.generate(topic)
        self._save_json("candidates.json", [dataclasses.asdict(c) for c in cands])
        if len(cands) < 2:
            return {"error": "not_enough_candidates"}
        # 2) Pairs
        budget = min(self.judge.cfg.pairwise_budget, (len(cands) * (len(cands) - 1)) // 2)
        pairs = await self._schedule_pairs(cands, budget)
        # 3) Judge pairs
        results = []
        for (ia, ib) in pairs:
            ca, cb = cands[ia], cands[ib]
            # Anonymous + randomized AB/BA already inside judge
            res = await self.judge.judge_pair(topic, ca, cb)
            results.append(res)
            self.rank.update(ca.cid, cb.cid, res["winrate_a"])
        self._save_json("pair_results.json", results)
        # 4) Final selection
        top = self.rank.top(cands, self.run.top_k)
        out = [
            {
                "elo": float(elo),
                "matches": int(n),
                "joke": c.text.strip(),
                "plan": c.plan.to_dict(),
                "cid": c.cid,
            }
            for (c, elo, n) in top
        ]
        meta["finished"] = now_iso()
        self._save_json("summary.json", {"meta": meta, "winners": out})
        return {"meta": meta, "winners": out}

    def _save_json(self, name: str, obj: Any) -> None:
        path = self.log_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


# -------------------------
# CLI
# -------------------------

def build_llm(provider: str) -> LLMClient:
    provider = provider.lower()
    if provider in {"hf", "huggingface"}:
        return HuggingFaceClient()
    if provider in {"openai", "oai"}:
        return OpenAIClient()
    raise SystemExit(
        f"Unknown provider '{provider}'. Implement your adapter by subclassing LLMClient."
    )




def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PlanSearch + LLM-as-a-judge for jokes")
    p.add_argument("--topic", type=str, required=True, help="Topic or context, e.g., 'penguins'")
    p.add_argument("--provider", type=str, default="hf", help="LLM provider adapter (hf|openai)")
    p.add_argument("--gen-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Generator model name (HF id or provider-specific)")
    p.add_argument(
        "--judge-models",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Comma-separated judge model names for jury aggregation",
    )
    p.add_argument("--num-candidates", type=int, default=24, help="Max joke plans to explore")
    p.add_argument("--beams", type=int, default=8, help="Beam size (final candidates)")
    p.add_argument("--jokes-per-plan", type=int, default=2, help="Materializations per plan")
    p.add_argument("--pairwise", type=int, default=120, help="Pairwise judging budget")
    p.add_argument("--top-k", type=int, default=5, help="How many winners to output")
    p.add_argument("--seed", type=int, default=7, help="Global RNG seed")
    p.add_argument("--outdir", type=Path, default=Path("runs"), help="Logs/output folder")
    # Novelty options
    p.add_argument("--novelty-corpus", type=str, default="", help="Path to corpus file/dir (comma-separated paths)")
    p.add_argument("--novelty-ngrams", type=str, default="3,4", help="n-gram sizes, e.g. '2,3,4'")
    p.add_argument("--novelty-max-corpus", type=int, default=10000, help="Max corpus items to load")
    p.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model id (e.g., sentence-transformers/all-MiniLM-L6-v2)")
    return p.parse_args(argv)


async def amain(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    llm = build_llm(args.provider)

    gen_cfg = GenConfig(
        model=args.gen_model,
        max_tokens=256,
        beams=args.beams,
        max_plans=args.num_candidates,
        jokes_per_plan=args.jokes_per_plan,
    )
    judge_cfg = JudgeConfig(
        models=[s.strip() for s in args.judge_models.split(",") if s.strip()],
        pairwise_budget=args.pairwise,
    )
    run_cfg = RunConfig(outdir=args.outdir, seed=args.seed, top_k=args.top_k)

    pipe = JokePipeline(llm, gen_cfg, judge_cfg, run_cfg)
    result = await pipe.run_topic(args.topic)

    winners = result.get("winners", [])
    if not winners:
        print("No winners produced.")
        return

    # Novelty evaluation (optional)
    novelty_info: Dict[str, Dict[str, Optional[float]]] = {}
    if args.novelty_corpus:
        paths = [Path(p.strip()) for p in args.novelty_corpus.split(",") if p.strip()]
        ngram_ns = [int(x) for x in args.novelty_ngrams.split(",") if x.strip().isdigit()]
        from_types = "HF" if args.provider in {"hf", "huggingface"} else "OpenAI"
        checker = NoveltyChecker(
            llm=llm if args.embedding_model else None,
            ngram_ns=ngram_ns or (3, 4),
            embedding_model=args.embedding_model or None,
            max_corpus=args.novelty_max_corpus,
        )
        checker.load_corpus(paths)
        for w in winners:
            txt = w["joke"]
            ng = checker.ngram_novelty(txt)
            emb = await checker.embedding_novelty(txt)
            novelty_info[w["cid"]] = {**ng, "embed_novelty": emb}
        with (pipe.log_dir / "novelty.json").open("w", encoding="utf-8") as f:
            json.dump(novelty_info, f, ensure_ascii=False, indent=2)

    print("=== Top Jokes ===")
    for i, w in enumerate(winners, 1):
        cid = w["cid"]
        extra = ""
        if cid in novelty_info:
            ni = novelty_info[cid]
            parts = []
            if isinstance(ni.get("ngram_avg"), (int, float)):
                parts.append(f"ngram {ni['ngram_avg']:.2f}")
            if isinstance(ni.get("embed_novelty"), (int, float)):
                parts.append(f"embed {ni['embed_novelty']:.2f}")
            if parts:
                extra = " | novelty " + ", ".join(parts)
        print(f"{i}) [Elo {w['elo']:.1f} | matches {w['matches']}{extra}]{w['joke']}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("Interrupted.")


if __name__ == "__main__":
    main()
