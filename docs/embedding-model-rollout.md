# Embedding model configuration

The default OpenAI model is `text-embedding-3-large`, explicitly requesting
1,536 dimensions for both `embed_text` and `embed_batch`. This preserves the
existing `vector(1536)` PostgreSQL storage and vector IVFFlat indexes. The
larger model's default 3,072-dimensional output is not used by this configuration.
Using a supported shorter output does not select the small embedding model.

Configure the pair together:

```text
OPENAI__EMBED_MODEL=openai:text-embedding-3-large
OPENAI__VECTOR_DIMENSION=1536
```

The provider uses these settings for queries, entity summaries, current and
historical fact fields, dictionary operations and new queued chat vectors.
Search text, ranking, thresholds, result counts and memory JSON are unchanged.
Existing thresholds were calibrated with the small model and need evaluation
before making broader retrieval-quality claims.

## Existing installations

Matching vector lengths do not make different models' vectors compatible.
Changing the configuration alone is not a migration. Bootstrap checks dimensions
but cannot identify which model produced an existing vector.

Stop host applications and workers before switching. Back up the database and
configuration. Rebuild every entity-summary and fact/history search vector from
the stored payloads, and rebuild dictionary anchors using their original input
evidence where available. An averaged dictionary anchor cannot be converted
mathematically to another model's space; its source texts must be re-embedded.
If original evidence is unavailable, choose an explicit dictionary reseeding
policy rather than silently treating the old vectors as compatible. Preserve
keyword IDs, labels, contexts and statistics when rebuilding an existing dictionary.

Redis may contain session vectors, queued message vectors and batches. Complete
old-model work before the switch or rebuild those vectors from retained texts.
Do not clear unrelated cache entries. Activate the new model only after all
persistent and operational vectors agree with it.

New, empty installations need no vector migration. Existing installations with
memories or dictionary entries need an explicit maintenance operation; there is
no automatic general-purpose model migration in the engine.

## Evaluation

Compare rankings on identical text, using a query and stored vectors generated
by the same model. Raw cosine scores are not directly calibrated across models.
Keep previous results and caches pinned to their original model and dimensions.
The engine evaluation budget supports both small and large embedding prices;
historical evaluation scripts with an explicit small-model assertion remain
reproduction tools, not automatic large-model reruns.

References:

- [OpenAI embedding dimensions](https://developers.openai.com/api/docs/guides/embeddings)
- [OpenAI large-model pricing](https://developers.openai.com/api/docs/models/text-embedding-3-large)
- [pgvector index dimension limits](https://github.com/pgvector/pgvector#ivfflat)
