# Why I built a RAG evaluator that admits what it can't judge

Most tools that score AI output have a quiet problem: they're an LLM grading an LLM, and they
report every number with the same confident tone — the ones they've actually verified and the ones
they've merely guessed. For evaluation, that's backwards. An evaluator's job is to be *more*
trustworthy than the thing it grades, not equally credulous.

`rag-triad` is a small, local evaluator for retrieval-augmented (RAG) answers built around one rule:
**lean on a deterministic check wherever one exists, and abstain — out loud — wherever one doesn't.**

## The problem it solves

A RAG answer can fail in three different places:

1. **Retrieval** fetched the wrong context.
2. **Generation** ignored the context and hallucinated.
3. The answer is grounded but **doesn't address the question.**

A single "quality: 0.72" score can't tell these apart — and they need completely different fixes.
So `rag-triad` scores the **RAG triad** (context relevance, groundedness, answer relevance) and
localizes *which* leg broke:

```
▸ fabricated citation
    context ✓ RELEVANT   grounded ✗ FAIL   answer ✓ RELEVANT
    → HALLUCINATION — the model left the context. Fix generation or enforce cite-and-verify.

▸ answer from the wrong context
    context ✗ IRRELEVANT grounded ✗ FAIL   answer ✓ RELEVANT
    → RETRIEVAL MISS — the right context wasn't retrieved. Fix chunking / embeddings / top-k.
```

Same shape of failure, two different diagnoses, two different fixes.

## What makes it trustworthy, not just another triad

The triad framing is standard (TruLens, RAGAS). The contribution here is the *discipline*:

- **Fail-closed groundedness.** The judge must cite an exact quote, and **code** — not the model —
  verifies that quote is actually in the context. A fabricated citation can't score as "grounded";
  the worst case is an honest DEFER, never a false pass.
- **A deterministic corroborator matched to each leg's failure mode.** Context relevance gets an
  embedding-similarity floor that overrides a mistaken "relevant" judge. Answer relevance gets an
  answer-*type* gate: if the question demands a number or a time the answer plainly lacks, it isn't
  relevant — no judge needed. (Reusing the embedding trick here would *backfire*, because cosine
  measures topical overlap and would happily bless a same-topic evasive answer. The signal has to
  fit the failure.)
- **Judges abstain instead of bluffing.** The model-judged legs sample several times; if they don't
  agree, the verdict is ABSTAIN — not a confident-but-meaningless score.
- **Validate the validator.** `--selftest` runs planted failures the evaluator *must* catch before
  you trust it: a fabricated citation (must fail groundedness), an honest refusal (must NOT read as
  a hallucination). An evaluator that's only ever seen clean inputs is unproven.

## Why I care about calibration specifically

I build small verification tools as a side practice, and one of them measures *when a model's stated
confidence is actually trustworthy*. A recurring failure in older models was being **confidently
wrong** on questions they couldn't actually compute — snapping to a plausible answer instead of
signaling uncertainty. Testing a newer-generation model recently, the failure mode had shifted: on
the same hard questions it *stopped emitting confident wrong answers* and instead ran out of room
trying to reason it through — inconclusive, but honest.

That shift is exactly what an evaluator should reward and what a naive scorer misses. A more capable
model that's confidently wrong is more dangerous than a weaker one that abstains. So `rag-triad` is
built to prize the honest "I can't tell" over the confident guess — because that's the property that
actually makes downstream AI safe to trust.

## Honest limits

- The relevance legs are model-judged; their reliability tracks the judge model you point them at.
- The embedding-floor threshold sits in a compressed range and is corpus-sensitive — tune it.
- The answer-type gate covers *typed* questions (number/time); open-ended "why/explain" falls back
  to the judge.

None of that is hidden. Knowing which of an evaluator's numbers you can bank and which you can't
*is* the product.

---

Code, demo, and design notes: [github.com/MonongahelaHellbender/rag-triad](https://github.com/MonongahelaHellbender/rag-triad).
Runs locally on Ollama — no API keys, no cloud. MIT licensed.
