from utils import * 

COMEDIC_DEVICES = [
    "misdirection",
    "benign_violation",
    "observational",
    "analogy",
    "rule_of_three",
    "anti_joke",
    "wordplay",
]

STYLES = ["deadpan", "dad_joke", "one_liner", "story", "tech_snark", "wholesome"]
AUDIENCES = ["general", "engineers", "parents", "students", "standup_crowd"]


@dataclass
class JokePlan:
    topic: str
    device: str
    style: str
    audience: str
    twist: str
    safety: str
    plan_id: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class JokeCandidate:
    plan: JokePlan
    text: str
    rationale: str
    cid: str


@dataclass
class JudgeConfig:
    models: List[str]
    temperature: float = 0.2
    top_p: float = 0.9
    pairwise_budget: int = 120
    anchors: int = 4
    ab_inversion: bool = True
    votes_per_pair: int = 2  # per model via seeds
    k_factor: float = 24.0  # Elo step size


@dataclass
class GenConfig:
    model: str
    temperature: float = 0.95
    top_p: float = 0.95
    min_p: Optional[float] = None  # if your provider supports it
    max_tokens: int = 256
    beams: int = 8
    max_plans: int = 24
    jokes_per_plan: int = 2


@dataclass
class RunConfig:
    outdir: Path
    seed: int = 7
    top_k: int = 5


# -------------------------
# Plan generation
# -------------------------

class PlanSearch:
    """Beam-style plan exploration with structured prompts.

    Why: Creative quality improves when generation is split into
    planning and realization. Beam exploration over devices/styles
    yields diverse joke structures before wording is chosen.
    """

    SYS_PLAN = (
        "You are a veteran comedy writer. You produce tight, safe, stage-ready joke plans "
        "for a given topic. Plans are concrete and avoid cliches."
    )

    PROMPT_PLAN = """
Topic: {topic}
You will propose a single joke plan as compact JSON with fields:
- device: one of {devices}
- style: one of {styles}
- audience: one of {audiences}
- twist: a specific unexpected angle that hinges on the device
- safety: constraints to keep it inoffensive and family-friendly
Only output JSON, no prose.
""".strip()

    SYS_WRITE = (
        "You are a comedy ghostwriter. Realize a joke plan as a short joke. "
        "Avoid slurs, strong profanity, politics, or punching down."
    )

    PROMPT_WRITE = """
Write {n} alternative jokes following this plan in JSON list form.
Each entry must have: "text" and brief "rationale" about why it should be funny.
Plan JSON:
{plan_json}
Length target: 1-3 sentences per joke.
Keep it clean; no explicit content.
""".strip()

    def __init__(self, llm: LLMClient, gen: GenConfig) -> None:
        self.llm = llm
        self.cfg = gen

    async def _one_plan(self, topic: str) -> JokePlan:
        device = random.choice(COMEDIC_DEVICES)
        style = random.choice(STYLES)
        audience = random.choice(AUDIENCES)
        plan_prompt = self.PROMPT_PLAN.format(
            topic=topic,
            devices=", ".join(COMEDIC_DEVICES),
            styles=", ".join(STYLES),
            audiences=", ".join(AUDIENCES),
        )
        out = await self.llm.acomplete(
            system=self.SYS_PLAN,
            messages=[{"role": "user", "content": plan_prompt}],
            model=self.cfg.model,
            temperature=0.8,
            top_p=self.cfg.top_p,
        )
        plan_json = self._force_json(out)
        plan = JokePlan(
            topic=topic,
            device=plan_json.get("device", device),
            style=plan_json.get("style", style),
            audience=plan_json.get("audience", audience),
            twist=plan_json.get("twist", f"unexpected angle on {topic}"),
            safety=plan_json.get("safety", "keep clean and inclusive"),
            plan_id=sha1(json.dumps(plan_json, sort_keys=True) + str(random.random())),
        )
        return plan

    def _force_json(self, text: str) -> Dict[str, Any]:
        """Be defensive: strip code fences/explanations and parse minimal JSON."""
        match = re.search(r"\{[\s\S]*\}", text)
        raw = match.group(0) if match else "{}"
        try:
            return json.loads(raw)
        except Exception:
            return {}

    async def _realize(self, plan: JokePlan, n: int) -> List[JokeCandidate]:
        plan_json = json.dumps(plan.to_dict(), ensure_ascii=False)
        out = await self.llm.acomplete(
            system=self.SYS_WRITE,
            messages=[
                {
                    "role": "user",
                    "content": self.PROMPT_WRITE.format(n=n, plan_json=plan_json),
                }
            ],
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            extra={"max_tokens": self.cfg.max_tokens},
        )
        jokes = []
        try:
            data = json.loads(re.search(r"\[[\s\S]*\]", out).group(0))  # type: ignore
        except Exception:
            data = []
        for obj in data:
            text = (obj.get("text") or "").strip()
            rationale = (obj.get("rationale") or "").strip()
            if not text:
                continue
            jokes.append(
                JokeCandidate(
                    plan=plan,
                    text=text,
                    rationale=rationale,
                    cid=sha1(plan.plan_id + text),
                )
            )
        return jokes

    def _pre_score(self, cand: JokeCandidate) -> float:
        """Cheap heuristic to prune unpromising candidates.

        Why: Saves judge budget by filtering clear duds while keeping diversity.
        """
        toks = simple_tokenize(cand.text)
        d2 = distinct_n(toks, 2)
        d3 = distinct_n(toks, 3)
        length_pen = -0.002 * max(0, len(cand.text) - 220)
        topical = 1.0 if cand.plan.topic.lower() in cand.text.lower() else 0.2
        return 0.6 * d2 + 0.3 * d3 + 0.1 * topical + length_pen

    @staticmethod
    def _dedup(cands: List[JokeCandidate], threshold: float = 0.92) -> List[JokeCandidate]:
        """Fuzzy string dedup using normalized Levenshtein ratio.

        Why: Prevent trivial paraphrase duplicates from flooding judging.
        """
        try:
            from rapidfuzz import fuzz  # type: ignore
        except Exception:
            fuzz = None

        kept: List[JokeCandidate] = []
        for c in cands:
            dup = False
            for k in kept:
                if fuzz is None:
                    if c.text == k.text:
                        dup = True
                        break
                else:
                    if fuzz.ratio(c.text, k.text) / 100.0 >= threshold:
                        dup = True
                        break
            if not dup:
                kept.append(c)
        return kept

    async def generate(self, topic: str) -> List[JokeCandidate]:
        plans = await asyncio.gather(*[self._one_plan(topic) for _ in range(self.cfg.max_plans)])
        jokes_nested = await asyncio.gather(*[self._realize(p, self.cfg.jokes_per_plan) for p in plans])
        cands = [c for lst in jokes_nested for c in lst]
        cands = self._dedup(cands)
        cands.sort(key=self._pre_score, reverse=True)
        return cands[: self.cfg.beams]
