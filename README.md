# rag-triad — a fail-closed, self-validating RAG evaluator

A small, dependency-light evaluator that scores a RAG answer on the **RAG triad** — context relevance,
groundedness, answer relevance — and, unusually, **tells you which of its three numbers you can trust.**
Runs locally on [Ollama](https://ollama.com); no API keys, no cloud.

The triad idea itself is standard (TruLens, RAGAS). What's different here is the **discipline**:

- **Fail-closed groundedness.** The model must cite a quote, and **code** — not the model — verifies the
  quote is really in the context. A fabricated citation can't score as "grounded"; the worst case is an
  honest DEFER.
- **Deterministic corroborators, matched to each leg's failure mode.** Context-relevance is backed by an
  embedding-similarity *floor* (a low value overrides a mistaken "relevant" judge); answer-relevance by an
  answer-*type* gate (if the question demands a number/time the answer lacks, it isn't relevant). The judge
  legs use self-consistency and **abstain** rather than emit a confident-but-worthless score.
- **Validate the validator.** `--selftest` runs planted failures the evaluator must catch before you trust
  it — a fabricated citation (must fail groundedness) and an honest refusal (must NOT be scored as a
  hallucination) among them.

## Run
```bash
ollama pull nomic-embed-text                 # embeddings
ollama pull qwen2.5-coder:7b                 # a judge model (any chat model works)
python3 rag_triad.py --selftest              # prove it catches planted failures
python3 rag_triad.py sample.json             # score one {"question","context","answer"} sample
```
Env: `TRIAD_MODEL` (judge model, default `llama3.2:3b`), `TRIAD_EMBED` (default `nomic-embed-text`),
`TRIAD_SAMPLES`, `TRIAD_RELEVANCE_FLOOR`.

## What it deliberately does NOT do
It doesn't pretend a judged number is certain. Groundedness is bankable (a deterministic gate); the
relevance legs are judged and will **abstain** when a small model can't be trusted. Knowing which is
which is the whole point — see [DESIGN.md](DESIGN.md).

## License
MIT © Melissa Ellison.
