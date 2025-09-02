
class BiasAwareJudge:
    """Pairwise judging with order randomization, AB/BA inversion and jury aggregation.

    Why: LLM-as-a-judge exhibits position and verbosity biases; we randomize and
    invert order, anonymize candidates, and aggregate across models/seeds to
    reduce systematic errors.
    """

    SYS_JUDGE = (
        "You are a neutral comedy critic. Judge only funniness for a general audience. "
        "Ignore length and formatting. Penalize cliches, meanness, or unsafe content."
    )

    PROMPT_PAIR = """
Evaluate which joke is funnier for a general audience **purely on humor**.
Do not reward length, formatting, or instruction following.
Return strict JSON: {{"winner": "A"|"B", "score": 1-10, "reason": "..."}}.

Topic/context: {topic}

Joke A:
{a}

Joke B:
{b}
""".strip()

    def __init__(self, llm: LLMClient, cfg: JudgeConfig) -> None:
        self.llm = llm
        self.cfg = cfg

    async def _vote_once(self, model: str, topic: str, a: str, b: str,
                          temperature: float, top_p: float) -> Tuple[str, float, str]:
        out = await self.llm.acomplete(
            system=self.SYS_JUDGE,
            messages=[
                {"role": "user", "content": self.PROMPT_PAIR.format(topic=topic, a=a, b=b)}
            ],
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        try:
            obj = json.loads(re.search(r"\{[\s\S]*\}", out).group(0))  # type: ignore
            winner = str(obj.get("winner", "A")).strip().upper()
            score = float(obj.get("score", 5))
            reason = str(obj.get("reason", "")).strip()
        except Exception:
            winner, score, reason = ("A", 5.0, "parse_error")
        if winner not in ("A", "B"):
            winner = "A"
        return winner, max(1.0, min(10.0, score)), reason

    async def judge_pair(self, topic: str, ca: JokeCandidate, cb: JokeCandidate) -> Dict[str, Any]:
        # AB
        tasks = []
        for m in self.cfg.models:
            for _ in range(self.cfg.votes_per_pair):
                tasks.append(
                    self._vote_once(
                        model=m,
                        topic=topic,
                        a=ca.text,
                        b=cb.text,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                    )
                )
        votes_ab = await asyncio.gather(*tasks)
        # BA inversion
        votes_ba: List[Tuple[str, float, str]] = []
        if self.cfg.ab_inversion:
            tasks = []
            for m in self.cfg.models:
                for _ in range(self.cfg.votes_per_pair):
                    tasks.append(
                        self._vote_once(
                            model=m,
                            topic=topic,
                            a=cb.text,
                            b=ca.text,
                            temperature=self.cfg.temperature,
                            top_p=self.cfg.top_p,
                        )
                    )
            votes_ba = await asyncio.gather(*tasks)

        # Aggregate with inversion correction
        wins_a = 0
        scores: List[float] = []
        rationales: List[str] = []
        for w, s, r in votes_ab:
            wins_a += 1 if w == "A" else 0
            scores.append(s)
            rationales.append(r)
        for w, s, r in votes_ba:
            # In BA, winner=="A" maps to B in original orientation
            wins_a += 0 if w == "A" else 1
            scores.append(s)
            rationales.append(r)
        total = len(votes_ab) + len(votes_ba)
        winrate_a = wins_a / max(1, total)
        return {
            "pair": (ca.cid, cb.cid),
            "winrate_a": winrate_a,
            "avg_score": sum(scores) / max(1, len(scores)),
            "votes": total,
            "notes": rationales[:5],
        }


class EloRanker:
    """Elo ranking from pairwise match results.

    Why: Simple, well-understood way to turn pairwise preferences
    into a global ranking with uncertainty approximations.
    """

    def __init__(self, k: float = 24.0) -> None:
        self.k = k
        self.R: Dict[str, float] = defaultdict(lambda: 1500.0)
        self.N: Dict[str, int] = Counter()

    @staticmethod
    def _expect(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, ida: str, idb: str, winrate_a: float) -> None:
        ra, rb = self.R[ida], self.R[idb]
        ea = self._expect(ra, rb)
        sa = winrate_a  # fractional score from aggregated votes
        self.R[ida] = ra + self.k * (sa - ea)
        self.R[idb] = rb + self.k * ((1 - sa) - (1 - ea))
        self.N[ida] += 1
        self.N[idb] += 1

    def top(self, items: List[JokeCandidate], k: int) -> List[Tuple[JokeCandidate, float, int]]:
        items_sorted = sorted(items, key=lambda c: self.R[c.cid], reverse=True)
        return [(c, self.R[c.cid], self.N[c.cid]) for c in items_sorted[:k]]
