#!/usr/bin/env python3
"""example.py — watch rag-triad diagnose three RAG answers in one command.

    TRIAD_MODEL=qwen2.5-coder:7b python3 example.py     # any Ollama chat model works as the judge

Each case is a (question, context, answer) triple. The evaluator scores the RAG triad and,
crucially, localizes *which* part failed:

    1. a grounded, on-topic answer      -> TRUSTWORTHY
    2. a fabricated citation            -> HALLUCINATION (caught by the deterministic gate)
    3. an answer from the wrong context -> RETRIEVAL MISS

Needs Ollama running with `nomic-embed-text` + a chat model.
"""
import rag_triad as t

CASES = [
    {"name": "grounded, on-topic answer",
     "question": "How long do I have to return an item?",
     "context": "Returns policy: items may be returned within 30 days of purchase with a valid receipt.",
     "answer": "You can return an item within 30 days of purchase, as long as you have the receipt."},
    {"name": "fabricated citation",
     "question": "How long do I have to return an item?",
     "context": "Returns policy: items may be returned within 30 days of purchase with a valid receipt.",
     "answer": "The policy states you have 90 days to return any item, and no receipt is required."},
    {"name": "answer from the wrong context",
     "question": "How do I reset my password?",
     "context": "Returns policy: items may be returned within 30 days of purchase with a valid receipt.",
     "answer": "Click 'Forgot password' on the sign-in page and follow the emailed link."},
]

MARK = {"PASS": "✓", "FAIL": "✗", "RELEVANT": "✓", "PARTIAL": "~", "IRRELEVANT": "✗",
        "N/A": "–", "ABSTAIN": "?"}


def main():
    print(f"rag-triad demo · judge={t.MODEL}\n" + "=" * 72)
    for c in CASES:
        try:
            r = t.evaluate(c)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[couldn't reach the judge/embedder — is Ollama running? {exc}]")
            return
        cr, g, ar = r["context_relevance"], r["groundedness"], r["answer_relevance"]
        print(f"\n▸ {c['name']}")
        print(f"    Q: {c['question']}")
        print(f"    A: {c['answer']}")
        print(f"    context {MARK.get(cr['verdict'],'?')} {cr['verdict']:10}   "
              f"grounded {MARK.get(g['verdict'],'?')} {g['verdict']:5}   "
              f"answer {MARK.get(ar['verdict'],'?')} {ar['verdict']}")
        print(f"    → {r['diagnosis']}")
    print("\n" + "=" * 72)
    print("A failing leg points at a different subsystem: retrieval, generation, or the prompt.")


if __name__ == "__main__":
    main()
