# Memory-quality evaluation plan

Status: Stage 1's first two development runs completed and were reviewed;
the remaining campaign is held while findings and evaluation scope are reviewed.
Stages 2 and 3 remain planned. See [Stage 1 preparation and budget](../evals/memory_quality/LONG_CAMPAIGN.md).
This plan revision does not start paid calls or change the engine.

## Retrieval purpose and evaluation revision

Tomte supplies whole entities as background memory for a conversation between
the user and a calling AI. The input can be an ordinary chat message, not a direct
request for a remembered answer. The caller uses that background to understand
the situation and chooses which details belong in its response.

The earlier fact-level answer-relevance criterion did not capture this purpose.
For example, a workplace conversation can reasonably receive job satisfaction,
colleagues and related concerns along with workplace addresses. Facts within a
relevant entity are not errors merely because they do not answer the latest
message. Whole-entity return remains intentional; this clarification does not
authorize fact filtering or a retrieval-code change.

For future evaluation, judge useful conversational background and coverage of
relevant entities. Distinguish useful context from genuinely off-topic entities,
wrong-subject attribution, unsupported memory and context that misleads the caller.
Measure payload size, latency and cost separately; bytes that are not needed for a
narrow answer are not automatically waste.

This is an explicit change of evaluation scope, not a retroactive passing score.
Keep the original frozen datasets, labels, 80% answer-relevant-fact target and
results in their historical evidence. The former 27.6% fact-level result does not
establish a background-memory retrieval defect. Retire that target for prospective
background-memory acceptance; freeze a versioned conversational rubric and any
new thresholds before new scoring. Do not silently relabel old runs or infer that
all returned entities are useful merely because whole entities are intended.

## What we know

The realistic pilot passed five authored conversations, each with eight user
messages: 40 messages, 25 consolidation batches and ten retrieval queries. It
tested similar subjects, ambiguity, correction versus change, mixed topics and
multilingual uncertainty. The deterministic suite passed 845 tests with
PostgreSQL and Redis enabled, zero failures/skips, and 21 warnings.

These are useful development results, not a general memory-accuracy estimate.
The live cases were short, used fresh dictionaries and small memory collections,
and received LLM-assisted manual review rather than blinded human adjudication.
Recent-memory fallback could help retrieval. Repeating these same cases would
measure stability on known examples, not performance on unseen conversations.

## Questions to answer

1. Does important information survive long, interrupted conversations?
2. Do changes affect the intended subject and fact while preserving valid history?
3. Does ambiguity cause safe abstention rather than a confident wrong memory?
4. Can retrieval find useful background entities for the current conversation among many plausible distractors?
5. Can a caller use that background appropriately and distinguish current information, past information and uncertainty?
6. How often do results vary, and what do ingestion and retrieval cost?

Wrong memories matter more than omissions. Report them separately; a high recall
score must not hide an unsupported claim or a wrong-person update. Python may
check structural invariants; natural-language judgments remain a review task,
not a new set of hardcoded language rules in the engine.

For this version, omission of a claim with an unresolved subject is an accepted
limitation. Do not require a pending/unattributed durable memory or weaken identity
checks to retain it. Report the lost information honestly, including whether it
was later supplied with a supported identity. The initial two ambiguous-brother
checkpoint misses remain in the historical scores; policy acceptance does not
retroactively make those memories present. Future rubrics must distinguish an
accepted owner-ambiguity omission from losing a supported, identified fact.

## Stage 1: longer conversations

Create six new synthetic conversations with 40 user messages each, spread across
eight simulated sessions of five messages. Target roughly 40-100 words per
English message, allowing natural short replies; record characters and model
tokens for all languages. Record actual word/token totals rather than assuming
equal message length. Include realistic assistant replies as untrusted context.

Use session timestamps spanning several simulated weeks. This tests temporal
reasoning and session boundaries; it does not prove weeks of real operation.
Start with fixed batches of three and two user messages per session: 16 batches
per conversation, 96 batches per complete pass. Persist memory between sessions,
but reset to isolated synthetic state between conversations and repetitions.

The six conversation themes are:

- Two jobs and two similar coworkers: employer details, concurrent employment,
  ending only one job, and a later new job.
- Similar possessions: two same-model cars, genuine nicknames, repairs, repainting
  and changes affecting only one car.
- Family and ambiguous references: two brothers, their workplaces, uncertain
  pronouns, explicit clarification and details that must remain unattributed.
- Home and logistics: corrected addresses, real moves, approximate dates,
  canceled plans and meaningful historical states.
- Mixed everyday topics: pets, classes, preferences and changing feelings,
  with repetition, irrelevant chatter and delayed returns to an earlier subject.
- Multilingual conversation: Chinese and English names, paraphrases, uncertainty
  followed by confirmation, qualified schedules and incomplete information.

Place important updates at early, middle and late positions. Separate a fact's
introduction from later references by at least ten user messages in selected
cases. Include multiple facts with the same attribute name and ensure the
scoring identifies instances by meaning and UUID, not attribute-name uniqueness.

Freeze eight memory checkpoints and eight retrieval queries per conversation:
four at intermediate sessions and four after the final session. Intermediate
checks must expose an earlier wrong attribution even if later corrected. Queries
cover current state, prior state, cross-fact questions, ambiguity and unknowns.
Queries themselves must not become assertions through consolidation.

Assign three conversations to development and three to held-out validation before
running. Balance the difficult behaviors across both groups. Finish tuning on
development cases before opening validation results. Run each frozen case three
times from fresh state: 18 planned runs, 720 user-message exposures, 288 batches
and 144 queries, excluding any explicitly recorded debugging runs. Start with
two development cases once before committing to the complete campaign.

If a validation failure leads to a fix, keep the failure and classify those cases
as known regression cases. Use new held-out cases for a fresh generalization
claim; never relabel the tuned result as an untouched validation result.

## Stage 2: retrieval beyond a small memory collection

Use two isolated synthetic memory collections: 100 and 500 entities, with a
declared distribution of facts and historical snapshots. Include near-matching
names, overlapping jobs, paraphrases, low-importance details, old targets and
plausible distractors. Deliberately put query targets outside recent-memory
fallback and recent session context.

Prepare 60 held-out conversation probes: 20 about current situations, 15 about
past situations, ten involving multiple subjects, ten with no stored background
for the topic and five ambiguous references. Include ordinary chat turns rather
than only explicit memory questions. Provide enough recent conversational context
to establish the topic, and freeze expected background entities, acceptable related
context and genuinely off-topic candidates before execution. Keep direct questions
as separately labeled diagnostic probes. Run the same set at each collection size
and report each tier separately; use separate development probes for tuning.

Directly seeded gold memories isolate retrieval performance and must be labeled
retrieval-only. Reuse selected Stage 1 memories separately to measure retrieval
after actual ingestion. Do not combine their scores. Compare cold dictionaries
with a frozen, populated synthetic dictionary in a declared follow-up; never
borrow the user's existing dictionary or treat arbitrary seeded data as evidence
of naturally learned vocabulary quality.

Report both API channels separately for diagnosis, and assess the complete context
delivered to the caller. The name `query_relevant_memories` does not mean every fact
inside an entity must answer the latest sentence. Background continuity can also
be useful. Missing a requested detail does not make related background invalid;
the caller must still distinguish what is known from what has not been supplied.
Do not require an empty response merely because a narrow answer is unavailable.

## Stage 3: caller comprehension

Use a fixed conversational-agent prompt, a current chat turn with bounded recent
context, and the public memory JSON. Do not provide gold, private metadata or the
full source conversation that created the memories. Check whether the agent uses
useful background naturally without forcing every retrieved detail into its reply,
confusing people, treating ended situations as current or inventing missing details.
Include a no-memory baseline on the same conversations. Retain direct current/
historical questions as a separate comprehension diagnostic, not the sole use case.

Score answers separately from storage and retrieval: a caller mistake must not
be silently counted as an extraction error, and a correct guess must not count
as a successful memory retrieval. Include a small human-reviewed sample before
relying on any automated judge. This stage has its own cost declaration.

## Scoring and proposed progression gates

Freeze meaning-based gold before model execution. For each checkpoint, identify
required, optional, unsupported and ambiguous propositions; supported entity/fact
groupings; relevant source messages; and allowed current/history interpretations.
Equivalent wording and valid alternative grouping pass. Exact field names are
not a semantic oracle. Check names and aliases as well as fact values.

Report these measures with numerators and denominators:

- Required-meaning recall: retained required meanings / expected required meanings,
  reported per checkpoint, conversation and run; optional details separately.
- Unsupported-memory rate: unsupported unique propositions / all stored unique
  propositions reviewed at a checkpoint. Report occurrence counts as well, so
  one persistent mistake is not confused with many independently invented claims.
- Identity errors: wrong-subject writes, accidental merges, duplicate subjects
  and wrong-fact updates, each reported separately.
- Update/history correctness: correctly applied expected updates / expected
  updates, including correction-versus-change and preservation of prior values.
- Background coverage: required background entities/meanings available to the
  caller / expected background entities/meanings for the conversational probe.
- Entity usefulness: useful returned background entities / reviewed returned
  entities, with ambiguous relevance judgments reported separately. Whole entities
  may contain details that are not needed for the latest sentence.
- Direct-question coverage, where tested: supported answer requirements present
  in delivered memory / expected answer requirements. Report this diagnostic
  separately from conversational background usefulness.
- Context size: returned bytes/tokens by channel; assess any caller harm separately
  rather than treating every non-answering fact as irrelevant.
- Stability: per-case passes out of three and the worst run, not only pooled means.
- Operational measures: ingestion/retrieval p50 and p95 latency, observed usage
  and estimated cost per message/query, model requests, repair/rejection counts,
  timeouts and budget stops. Do not equate estimated cost with an invoice.

Proposed gates to freeze before execution:

- Zero tenant/user leakage, unauthorized UUID writes, history corruption or
  replay duplication. Any occurrence stops the affected run for investigation.
- Zero observed unsupported assertions or wrong-subject writes in held-out
  validation. An unresolved judgment is not a pass. This is an acceptance gate
  for the finite sample, not a claim that the true error rate is zero.
- At least 95% pooled required-meaning recall, with no conversation-run below
  90%; at least 95% of expected updates/history transitions correct.
- Freeze conversation-level background coverage, entity usefulness and caller-use
  criteria before the revised retrieval evaluation. No replacement numerical target
  is claimed as agreed or achieved here. The original answer-only thresholds remain
  in historical run evidence and must not be used to penalize useful background.
- No observed wrong-person use, invented facts or confusion between current and
  ended situations in the caller sample. Report natural background use and
  no-memory comparisons separately from direct-question diagnostic scores.

These thresholds are proposed engineering targets, not previously achieved
results or production guarantees. Do not move thresholds after seeing outcomes
within a frozen evaluation. An explicit product-scope revision, such as the
background-memory clarification above, requires a separately versioned rubric and
preservation of the earlier measurements; it cannot convert an old failure to a pass.
Avoid one blended accuracy number. With only a few authored conversations,
repetitions and facts are correlated; do not treat them as independent evidence
for a population confidence interval.

## Execution, diagnosis and evidence

Before each stage, pin the engine commit, prompts, dataset/gold hashes, provider
and model identifiers, settings, dictionary snapshot, package versions and
evaluator version. A published baseline should come from a reproducible checkout;
record any intentional patches rather than silently running a dirty tree.

The runner now validates long datasets and its executor supports session handling
and incremental queries. A separate campaign runner enforces persistent budget
accounting and phase review gates; spending authorization is still required.
Preserve old datasets and result directories.
Keep gold unavailable to engine calls. Save raw provider outputs, all checkpoint
snapshots, public query responses, timing, cost and cleanup evidence privately.
Publish only reviewed synthetic fixtures, methodology and selected results.

Review every stored proposition, including extra facts the gold did not request.
Use source-message and snapshot pointers for judgments. An LLM judge can assist,
but ambiguous cases and every suspected wrong memory require manual adjudication.
Record reviewer disagreements and the limits of LLM-assisted review.

For each failure, identify the first incorrect stage: extraction, identity,
dictionary alignment, mutation, validation/recovery, persistence, retrieval or
caller interpretation. Preserve the original output and add an offline replay
when practical. A model omission is a missed-memory result; a Python rejection
is not automatically a bug. Do not weaken identity or ownership validation to
improve recall. Fix one confirmed issue, rerun that case, then its related group;
run a full deterministic regression before meaningful integration or release.

All live work requires an estimate from current rates and expected token sizes,
an enforceable total campaign ceiling, request/time limits and explicit stop
rules. Estimate the whole planned campaign, not only its first batch. Do not split
a larger campaign into artificial tasks to bypass spending approval. Do not run
with unpriced models or unknown usage. Include retries and answering/judge calls
in the budget; stop future calls when the authorized boundary is reached.

Use new, scoped test databases/schemas, synthetic users and isolated Redis keys.
Verify protected-state digests before and after; clean up only owned test state.
Docker must be available. Do not reset existing memories, dictionaries or queues.

## Next evaluation preparation

Preserve Stage 1's frozen conversations, gold and observed outcomes. Complete
review of the remaining memory findings and record a versioned conversational
background rubric before new retrieval scoring. Continue within the existing
campaign's cumulative budget and review gates when execution is authorized;
this document update itself does not resume the campaign. Stages 2 and 3 remain
separate planned work after Stage 1 review.
