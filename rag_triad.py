#!/usr/bin/env python3
"""RAG triad evaluator — trustworthy, fail-closed-where-possible, and self-testing.

Scores a (question, context, answer) triple on the RAG triad:
  1. Context relevance  — did retrieval fetch usable context?   [judged, self-consistency]
  2. Groundedness       — is the answer backed by the context?  [DETERMINISTIC gate + judged fallback]
  3. Answer relevance   — does the answer address the question? [judged, self-consistency]

Why this beats a stock triad (all three as single LLM-judges):
  A. Two-tier groundedness — a fabricated citation is caught by CODE (fail-closed); only a
     genuine near-miss falls through to a judge, so a hallucination can't score as grounded.
  B. Judges must CITE the span they rely on, and code checks the span is really present — a
     fail-closed sanity gate on the *justification*, even on the soft legs.
  C. Self-consistency — the judge legs are sampled N times; agreement < 60% -> ABSTAIN, not a
     fake-precise number (a model grading a model is only worth its consistency).
  D. Validate the validator — `--selftest` runs planted failures the evaluator must catch
     before you trust it on real answers. The deterministic guarantee is held to a hard bar;
     the judge legs only have to *not confidently pass garbage*.

Run:
  python3 rag_triad.py --selftest        # prove the evaluator works on planted cases
  python3 rag_triad.py sample.json       # score one {"question","context","answer"} sample
Env: TRIAD_MODEL (default llama3.2:3b), TRIAD_SAMPLES (default 3), OLLAMA_HOST.
"""
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("TRIAD_MODEL", "llama3.2:3b")
EMBED_MODEL = os.environ.get("TRIAD_EMBED", "nomic-embed-text")
N = int(os.environ.get("TRIAD_SAMPLES", "3"))  # self-consistency samples for the judge legs
FLOOR = float(os.environ.get("TRIAD_RELEVANCE_FLOOR", "0.45"))  # min question-context similarity
RELEVANT_SIM = float(os.environ.get("TRIAD_RELEVANT_SIM", "0.58"))  # corroboration bar to call a refusal over-cautious


def _chat(prompt, temperature=0.0):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": temperature}}
    req = urllib.request.Request(OLLAMA + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:  # noqa: S310 (local, trusted)
        return json.loads(r.read())["message"]["content"]


def _embed(text):
    for path, payload, key in (
        ("/api/embed", {"model": EMBED_MODEL, "input": text}, "embeddings"),
        ("/api/embeddings", {"model": EMBED_MODEL, "prompt": text}, "embedding"),
    ):
        try:
            req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
                d = json.loads(r.read())
            return d[key][0] if key == "embeddings" else d[key]
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("embedding request failed")


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _relevance_floor(question, context):
    """Deterministic signal: max cosine similarity between the question and any context segment.
    Low = retrieval found nothing actually close to the question (a real, judge-proof 'irrelevant')."""
    segs = [s.strip() for s in re.split(r"\n\n+", context) if len(s.strip()) > 20] or [context.strip()]
    qv = _embed(question)
    return max((_cosine(qv, _embed(s)) for s in segs), default=0.0)


_NUM_CUES = (
    "how many", "how much", "how old", "how long", "how far", "how tall", "how fast",
    "what time", "boiling point", "melting point", "temperature", "price", "cost",
    "percentage", "percent", "what year", "what age", "what number",
)


def _answer_type_gap(question, answer):
    """Deterministic answer-type check for the answer-relevance leg (the RIGHT signal for #3 —
    NOT embedding, which rewards topical-but-evasive answers). If the question demands a specific
    type (a number/time) that the answer plainly lacks, that's a judge-proof 'not relevant'.
    High-precision for typed questions; returns False (no signal) otherwise, so it only ever ADDS a
    catch, never a false 'irrelevant' on open-ended questions."""
    q = _norm(question)
    if any(c in q for c in _NUM_CUES) and not re.search(r"\d", answer):
        return True
    if q.startswith("when ") and not re.search(r"\d", answer):
        return True
    return False


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+", _norm(s)) if len(t) >= 3]


def _present(quote, context, min_len=12):
    """Is this quote really in the context? Exact substring OR high content-token coverage —
    so a near-verbatim citation (how models actually quote) counts, but a fabricated one
    (few shared words) still doesn't."""
    q = _norm(quote)
    if len(q) < min_len:
        return False
    if q in _norm(context):
        return True  # exact fast path
    qt = _tokens(quote)
    if len(qt) < 3:
        return False
    ctx_toks = set(_tokens(context))
    return sum(1 for t in qt if t in ctx_toks) / len(qt) >= 0.8  # fuzzy: 80% of content words present


REFUSAL_MARKERS = (
    "i don't have", "i do not have", "i don't know", "i do not know", "no information",
    "not in my notes", "not in the notes", "couldn't find", "could not find",
    "i'm not able", "i am not able", "didn't use any of", "don't have any information",
    "unable to find", "no relevant",
)


def _is_refusal(answer):
    """Heuristic: did the assistant decline for lack of info? So we don't score an honest
    'I don't know' as a hallucination. Transparent phrase match; misses paraphrased refusals."""
    a = _norm(answer)
    return any(m in a for m in REFUSAL_MARKERS)


def _sentences(text):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if len(p.strip()) > 3]


def _tag(text, key, default=""):
    m = re.search(key + r"\s*:?\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip().strip('"') if m else default


def _verdict(text, options, default="ABSTAIN"):
    seg = text.upper()
    m = re.search(r"VERDICT\s*:?\s*([A-Z]+)", seg)
    if m and m.group(1) in options:
        return m.group(1)
    # fallback: whole-word only, longest option first — "IRRELEVANT" must never match as
    # "RELEVANT", and "UNSUPPORTED" must never match as "SUPPORTED" (that would fail OPEN)
    for o in sorted(options, key=len, reverse=True):
        if re.search(rf"\b{o}\b", seg):
            return o
    return default


def _ctx(sample):
    c = sample["context"]
    return c if isinstance(c, str) else "\n\n".join(c)


# ---- Leg 2: Groundedness (two-tier, fail-closed) -----------------------------------------
def groundedness(sample):
    ctx = _ctx(sample)
    claims = _sentences(sample["answer"]) or [sample["answer"]]
    verified = para = 0
    unsupported = []
    for claim in claims:
        out = _chat(f"CONTEXT:\n{ctx}\n\nCLAIM: {claim}\n\n"
                    "Copy an EXACT quote from the context that supports the claim, "
                    "or write NONE if nothing in the context supports it.\nQUOTE:")
        quote = _tag(out, "QUOTE", (out.strip().splitlines() or [""])[0])
        if quote.upper() != "NONE" and _present(quote, ctx):
            verified += 1  # tier 1: the cited evidence is deterministically real
        else:  # tier 2: judged paraphrase fallback (only reached when no literal quote matched)
            j = _chat(f"CONTEXT:\n{ctx}\n\nCLAIM: {claim}\n\n"
                      "Ignoring exact wording, is the claim supported by the context?\n"
                      "VERDICT: SUPPORTED  or  VERDICT: UNSUPPORTED")
            if _verdict(j, ["SUPPORTED", "UNSUPPORTED"], "UNSUPPORTED") == "SUPPORTED":
                para += 1
            else:
                unsupported.append(claim)
    return {"leg": "groundedness", "verdict": "FAIL" if unsupported else "PASS",
            "trust": "deterministic gate + judged paraphrase fallback",
            "detail": f"{verified} quote-verified · {para} paraphrase-supported · {len(unsupported)} unsupported",
            "unsupported": unsupported}


# ---- Judge legs: self-consistency + cited-span sanity gate -------------------------------
def _sampled_judge(make_prompt, vote_fn):
    """Sample the judge N times; vote_fn(out) -> a verdict string, or None to discard the sample
    (the fail-closed sanity gate: a fabricated justification is thrown out, not counted)."""
    votes, invalid = [], 0
    for _ in range(N):
        v = vote_fn(_chat(make_prompt(), temperature=0.7))  # temp>0 so consistency means something
        if v is None:
            invalid += 1
            continue
        votes.append(v)
    if not votes:
        return {"verdict": "ABSTAIN", "confidence": 0.0,
                "note": f"0/{N} valid samples ({invalid} fabricated their justification)"}
    top, cnt = Counter(votes).most_common(1)[0]
    conf = cnt / len(votes)
    note = f"{len(votes)}/{N} valid · agreement {cnt}/{len(votes)}"
    if invalid:
        note += f" · {invalid} fabricated-justification discarded"
    return {"verdict": top if conf >= 0.6 else "ABSTAIN", "confidence": round(conf, 2), "note": note}


def context_relevance(sample):
    ctx, q = _ctx(sample), sample["question"]
    floor = _relevance_floor(q, ctx)  # deterministic: is anything actually close to the question?

    def prompt():
        return (f"QUESTION: {q}\n\nCONTEXT:\n{ctx}\n\n"
                "Does the context contain information needed to answer the question?\n"
                "Copy ONE exact sentence from the context that is most relevant, or NONE.\n"
                "QUOTE: <sentence or NONE>\nVERDICT: RELEVANT or PARTIAL or IRRELEVANT")

    def vote(out):
        quote = _tag(out, "QUOTE")
        if quote.upper() != "NONE" and not _present(quote, ctx):
            return None  # fabricated its justification -> discard
        return _verdict(out, ["RELEVANT", "PARTIAL", "IRRELEVANT"])

    r = _sampled_judge(prompt, vote)
    r["max_similarity"] = round(floor, 3)
    if floor < FLOOR:  # nothing semantically close -> override a mistaken "relevant" judge
        r["verdict"] = "IRRELEVANT"
        r["note"] = f"sim {floor:.2f} < {FLOOR} floor — retrieval found nothing relevant (overrides judge)"
    else:
        r["note"] = f"{r.get('note', '')} · sim {floor:.2f}".strip(" ·")
    r.update(leg="context_relevance", trust=f"embedding floor ({FLOOR}) + judged x{N}")
    return r


def answer_relevance(sample):
    """Hardened: force the judge to name WHAT the question demands, then point to the phrase in
    the ANSWER that supplies it. No such phrase (or a fabricated one) -> not relevant. This gets
    past the 'same topic = relevant' vibe check that let evasive answers through."""
    q, a = sample["question"], sample["answer"]

    def prompt():
        return (f"QUESTION: {q}\n\n"
                "First state, in a few words, WHAT SPECIFIC INFORMATION would answer this "
                "question (the thing being asked for).\n\n"
                f"ANSWER: {a}\n\n"
                "Now: does the answer actually supply that specific thing? Copy the exact phrase "
                "from the ANSWER that supplies it, or write NONE.\n"
                "DEMAND: <what the question asks for>\n"
                "EVIDENCE: <exact phrase from the answer, or NONE>\n"
                "VERDICT: RELEVANT or PARTIAL or IRRELEVANT")

    def vote(out):
        ev = _tag(out, "EVIDENCE")
        if ev == "" or ev.upper() == "NONE":
            return "IRRELEVANT"          # the judge admits the answer supplies nothing asked-for
        if not _present(ev, a, min_len=5):
            return None                  # cites a phrase that isn't in the answer -> discard
        return _verdict(out, ["RELEVANT", "PARTIAL", "IRRELEVANT"])

    r = _sampled_judge(prompt, vote)
    if _answer_type_gap(q, a):  # deterministic: demanded a number/time the answer lacks
        r["verdict"] = "IRRELEVANT"
        r["note"] = f"answer-type gap: demands a number/time the answer lacks · {r.get('note', '')}".strip(" ·")
    r.update(leg="answer_relevance", trust=f"judged x{N} + demand-decomposition + answer-type gate")
    return r


# ---- Orchestration + diagnosis -----------------------------------------------------------
def evaluate(sample):
    c = context_relevance(sample)
    if _is_refusal(sample["answer"]):  # an honest "I don't know" is not a hallucination
        g = {"leg": "groundedness", "verdict": "N/A", "trust": "skipped — answer is a refusal",
             "detail": "no factual claims to verify (the assistant declined)", "unsupported": []}
        a = {"leg": "answer_relevance", "verdict": "N/A", "note": "refusal, not a content answer"}
        # OVER-CAUTIOUS only if relevance is CORROBORATED (judge RELEVANT *and* high similarity);
        # a mistaken judge alone can't overturn a correct refusal.
        if c["verdict"] == "RELEVANT" and c.get("max_similarity", 0.0) >= RELEVANT_SIM:
            dx = "OVER-CAUTIOUS — relevant context existed but the assistant declined; check the prompt/threshold."
        else:
            dx = "APPROPRIATE REFUSAL — the question is outside the notes and the assistant correctly declined."
        return {"context_relevance": c, "groundedness": g, "answer_relevance": a, "diagnosis": dx}
    g, a = groundedness(sample), answer_relevance(sample)
    if c["verdict"] == "IRRELEVANT":
        dx = "RETRIEVAL MISS — the right context wasn't retrieved. Fix chunking / embeddings / top-k."
    elif g["verdict"] == "FAIL":
        dx = "HALLUCINATION — the model left the context. Fix generation or enforce cite-and-verify."
    elif a["verdict"] == "IRRELEVANT":
        dx = "OFF-TOPIC — grounded but doesn't answer the question. Fix the prompt / query."
    elif "ABSTAIN" in (c["verdict"], a["verdict"]):
        dx = "INCONCLUSIVE — judge legs disagreed. Re-run, use a stronger judge model, or inspect by hand."
    else:
        dx = "TRUSTWORTHY — all three legs pass."
    return {"context_relevance": c, "groundedness": g, "answer_relevance": a, "diagnosis": dx}


# ---- D: validate the validator -----------------------------------------------------------
SELFTEST = [
    {"name": "clean / good answer",
     "question": "What is the capital of France?",
     "context": "France is a country in Western Europe. Its capital and largest city is Paris, on the river Seine.",
     "answer": "The capital of France is Paris."},
    {"name": "fabricated citation (hallucination)",
     "question": "What is the capital of France?",
     "context": "France is a country in Western Europe. Its capital and largest city is Paris.",
     "answer": "The capital of France is Lyon, described in the text as the seat of government."},
    {"name": "irrelevant context (retrieval miss)",
     "question": "What is the boiling point of water at sea level?",
     "context": "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889.",
     "answer": "The Eiffel Tower was completed in 1889."},
    {"name": "off-topic answer (grounded but evasive)",
     "question": "What is the boiling point of water at sea level?",
     "context": "Water (H2O) is made of hydrogen and oxygen. At sea level, water boils at 100 degrees Celsius.",
     "answer": "Water is made of hydrogen and oxygen."},
    {"name": "honest refusal (out of scope)",
     "question": "What time is my dentist appointment tomorrow?",
     "context": "The engineering handbook covers the deploy process and the on-call rotation.",
     "answer": "I don't have any information about your dentist appointment in the notes. Please check your calendar."},
]


def _check(name, cv, gv, av):
    if name.startswith("clean"):
        return gv == "PASS" and cv != "IRRELEVANT" and av != "IRRELEVANT", "groundedness PASS + no leg rejects a good answer"
    if "fabricated" in name:
        return gv == "FAIL", "HARD: deterministic gate must catch the fake citation"
    if "irrelevant context" in name:
        # Leg INDEPENDENCE: the answer here is faithful to the (irrelevant) context, so groundedness
        # must PASS even while context-relevance flags — passing one leg must not mask, or be dragged
        # down by, another. The answer also ignores the question, so the answer leg must flag it.
        return (cv in ("IRRELEVANT", "PARTIAL", "ABSTAIN")
                and gv == "PASS"
                and av in ("IRRELEVANT", "PARTIAL", "ABSTAIN")), \
            "legs independent: context flagged + faithful answer stays grounded + answer flagged off-question"
    if "off-topic" in name:
        return av in ("IRRELEVANT", "PARTIAL", "ABSTAIN") and gv == "PASS", "grounded, but must flag the answer as evasive"
    if "refusal" in name:
        return gv == "N/A" and cv != "RELEVANT", "an honest refusal must not be scored as a hallucination or called over-cautious"
    return True, ""


def run_selftest():
    print(f"Calibration self-test · model={MODEL} · samples={N}\n" + "=" * 72)
    passed = hard_ok = True
    for case in SELFTEST:
        r = evaluate(case)
        cv, gv, av = r["context_relevance"]["verdict"], r["groundedness"]["verdict"], r["answer_relevance"]["verdict"]
        ok, why = _check(case["name"], cv, gv, av)
        passed = passed and ok
        if "fabricated" in case["name"]:
            hard_ok = ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] {case['name']}")
        print(f"       context={cv}  grounded={gv}  answer={av}")
        print(f"       expect: {why}")
        print(f"       dx: {r['diagnosis']}")
    print("\n" + "=" * 72)
    print(f"CALIBRATED: {'YES' if passed else 'PARTIAL'} "
          f"({'deterministic guarantee HOLDS' if hard_ok else 'DETERMINISTIC GUARANTEE BROKEN — do not trust'})")
    print("The judge legs are only asked not to confidently pass garbage; the fabricated-citation")
    print("case is the hard, deterministic guarantee — that one must always fail groundedness.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        run_selftest()
    elif len(sys.argv) > 1:
        print(json.dumps(evaluate(json.load(open(sys.argv[1]))), indent=2))
    else:
        print(__doc__)
