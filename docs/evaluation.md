# Evaluation evidence and limits

These are historical observations on synthetic data, not a general accuracy
claim or a fresh benchmark of this release candidate. Both studies used the
same archived private study snapshot.
Its Git history is not part of the fresh public v1.0.0 snapshot.

## TCX-027: conversation to durable memory

Three synthetic users, nine sessions, 24 user turns and 24 host-authored assistant
replies went through real providers and local PostgreSQL/Redis. Of seven target
current states, **5 were correct, 1 partial and 1 failed**. Of ten authored history
units, 6 were complete, 3 partial and 1 absent. This tests extraction and mutation,
not just retrieval. Assistant replies were frozen, not generated as an extra
chat-model experiment. Fictional dates were not a real multi-day TTL test.

## TCX-028: seeded retrieval

A directly seeded corpus of 12 synthetic entities/60 assertions was queried with
48 frozen questions: 40 answerable and 8 unknown. Real embeddings were used;
there was no extraction or consolidation in this study. Raw semantic Hit@1 was
**39/40**, and mean entity Recall@5 **40/40**; full entity coverage was 44/44
entity-query pairs. Required payload facts were present 55/55; rendered current
facts 48/48, but history rendering **0/7** because current-only projection omitted
non-active assertions. Unknown queries supplied nonempty context 8/8. These are
retrieval/presentation metrics, not answer correctness or a universal success rate.

At that checkpoint, background occupied early display positions. The subsequent
public two-list query/background contract has **not been rebenchmarked**. Its
expected ordering benefit is not a measured improvement. The v1.0.0 candidate
uses a later source snapshot, not the study snapshot.

## Reproduction and provenance

The study identifiers above refer to private development reports; private Drive
links and user data are intentionally not distributed. Public source includes
synthetic prompt fixtures under `evals/`, deterministic tests, and evaluation
scripts. `scripts/evaluate_retrieval_stack.py` and `evaluate_memory_ordering.py`
are historical comparisons: their ablation orders/metrics are retained, while
current context assertions now check the public two-list representation. The
captured query stream can include current-message lexical hits; do not relabel
new captured measurements as pure SQL cosine results.

Some historical harnesses require ignored embedding caches and historical
artifacts not distributed here. They are not turnkey reproductions of the exact
TCX-027/028 runs. Reproduction of those scores remains a release evidence gap;
a fresh, frozen, distributable corpus and authorized paid run would be separate
work. Do not regenerate labels, infer missing scores, or claim independent human
validation. Small synthetic corpora, generated language, one-off runs, model
nondeterminism and no large-catalog testing limit generalization.

Run `uv run pytest tests/test_evaluation_context_contract.py` for offline formatter
contract checks; this is not an IR benchmark. Live evaluator scripts can incur
API costs and write isolated synthetic fixtures before scoped cleanup. Review
them and their prerequisites before opting in. No live evaluation ran as part
of release hardening.
