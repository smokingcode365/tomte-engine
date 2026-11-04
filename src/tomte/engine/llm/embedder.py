"""Search text contains business data only; every current/history item is indexed."""
import json
import math
from tomte.engine.core.memory import public_value

SEARCH_TEXT_CHARS = 2000
EMBED_BATCH_SIZE = 16


def get_entity_summary(entity_data):
    identity = {key: entity_data.get(key) for key in ("entity_name", "entity_type", "entity_domain", "aliases")}
    # This bounded aggregate is a discovery fallback. Complete value chunks are
    # indexed separately; truncating this summary never discards stored memory.
    identity["facts"] = [{"attribute_name": f["attribute_name"], "value": public_value(f["value"])}
                         for f in entity_data.get("facts", [])]
    return json.dumps(identity, ensure_ascii=False)[:SEARCH_TEXT_CHARS]


def get_search_entries(entity_data):
    """Stable positional keys reference authoritative JSON, never replace it."""
    entries = []
    identity = json.dumps({key: entity_data[key] for key in ("entity_name", "entity_type", "entity_domain")}, ensure_ascii=False)
    for fact in entity_data["facts"]:
        for history_index, value in [(-1, fact["value"]), *enumerate(h["value"] for h in fact["history_values"])]:
            prefix = identity[:400] + " " + fact["attribute_name"][:200] + (" current " if history_index == -1 else " history ")
            for field_name, item in value.items():
                text = field_name + ": " + item["text"] + " (fact_confidence: " + item["fact_confidence"] + ")"
                for chunk_index, start in enumerate(range(0, len(text), SEARCH_TEXT_CHARS - len(prefix))):
                    entries.append({"fact_id": fact["fact_id"], "history_index": history_index,
                                    "field_name": field_name, "chunk_index": chunk_index,
                                    "text": prefix + text[start:start + SEARCH_TEXT_CHARS - len(prefix)]})
    return entries


def search_entry_key(entry):
    return (entry["fact_id"], entry["history_index"], entry["field_name"], entry["chunk_index"])


def changed_search_entries(entity_data, previous=None):
    old = {search_entry_key(entry): entry["text"] for entry in get_search_entries(previous)} if previous else {}
    return [entry for entry in get_search_entries(entity_data) if old.get(search_entry_key(entry)) != entry["text"]]


def embed_texts(llm, texts):
    vectors = []
    dimension = llm.get_vector_dimension()
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        result = llm.embed_batch(batch)
        if len(result) != len(batch):
            raise ValueError("Embedding response count mismatch")
        for vector in result:
            if (len(vector) != dimension or any(type(x) not in (int, float) or not math.isfinite(x) for x in vector)
                    or not any(vector)):
                raise ValueError("Invalid search embedding")
        vectors.extend(result)
    return vectors


def embed_entity(llm, entity_data):
    return embed_texts(llm, [get_entity_summary(entity_data)])[0]
