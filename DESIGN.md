# Design notes — why this evaluator is built the way it is

The RAG triad (context relevance / groundedness / answer relevance) is standard. The value here is a set
of design choices that make the evaluator **trustworthy about its own limits.**

## 1. Reduce dependence on the LLM judge
An LLM grading an LLM is only as good as the grader. So every leg leans on a **deterministic corroborator
matched to its actual failure mode** — not one generic trick:

- **Context relevance** — an embedding-similarity *floor*. If nothing in the context is close to the
  question, that's a judge-proof "retrieval miss" that overrides a mistaken "relevant" judge.
- **Groundedness** — the cited quote must be *present* in the context (fuzzy-matched to allow near-verbatim
  citation). Fail-closed: a fabricated citation can't pass.
- **Answer relevance** — an answer-*type* gate. If the question demands a number/time the answer lacks, it
  isn't relevant — no judge needed. *(Reusing the embedding trick here would backfire: cosine measures
  topical overlap, so it rewards topical-but-evasive answers. The signal must fit the failure mode.)*

## 2. Judges abstain instead of bluffing
The judge legs sample N times; agreement below a threshold → **ABSTAIN**, never a fake-precise score. A more
capable model that is *confidently wrong* is worse than a weaker one that abstains — so the evaluator is
built to say "I can't tell" rather than guess.

## 3. Validate the validator
`--selftest` runs planted failures the evaluator must catch. The fabricated-citation case is a hard,
deterministic guarantee (must fail groundedness); the refusal case ensures an honest "I don't know" is
never scored as a hallucination. An evaluator that has only seen clean cases is unproven.

## Honest limits
- The relevance legs are judged; their reliability tracks the judge model.
- The embedding-floor threshold is corpus-sensitive (embedding similarities sit in a compressed range) —
  tune `TRIAD_RELEVANCE_FLOOR`.
- The answer-type gate covers *typed* questions (number/time); open-ended "why/explain" falls back to the judge.

## Prior art
The triad framing follows **TruLens** and **RAGAS**. The contribution here is the fail-closed +
deterministic-corroborator + validate-the-validator discipline layered on top — not the triad itself.
