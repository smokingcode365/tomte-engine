"""Isolate invalid LLM proposal items; never relax the stored-memory contract."""
from copy import deepcopy
from typing import ClassVar

from pydantic import PrivateAttr, ValidationError, model_validator

from tomte.engine.core.memory import MemoryModel


def at_path(value, path):
    for part in path:
        value = value[part]
    return value


class ProposalResponse(MemoryModel):
    """Typed envelope with private rejected-item evidence for one bounded repair.

    Collections optionally contain another item list (entities/facts or plans/
    operations). Only a whole declared item can be discarded. A malformed envelope
    still raises; field-level salvage would change replacement semantics.
    """
    proposal_collections: ClassVar[dict[str, str | None]] = {}
    _rejected: list[tuple] = PrivateAttr(default_factory=list)
    _raw: dict = PrivateAttr(default_factory=dict)
    _validation_errors: list[dict] = PrivateAttr(default_factory=list)
    _locations: dict[tuple, tuple] = PrivateAttr(default_factory=dict)
    _repair_attempted: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def isolate_items(cls, data, handler):
        if isinstance(data, cls):
            return handler(data)
        original = deepcopy(data)
        rejected = set()
        errors = []
        try:
            result = handler(deepcopy(data))
        except ValidationError as exc:
            if not isinstance(data, dict):
                raise
            errors = exc.errors(include_input=False, include_context=False, include_url=False)
            for error in errors:
                path = error["loc"]
                if (len(path) < 2 or path[0] not in cls.proposal_collections
                        or type(path[1]) is not int):
                    raise  # The response envelope itself is unusable.
                nested = cls.proposal_collections[path[0]]
                end = 4 if len(path) >= 4 and path[2] == nested and type(path[3]) is int else 2
                rejected.add(path[:end])
            # A rejected parent subsumes all its rejected children.
            rejected = {p for p in rejected if p[:2] == p or p[:2] not in rejected}
            cleaned = deepcopy(data)
            for path in sorted(rejected, reverse=True):
                del at_path(cleaned, path[:-1])[path[-1]]
            result = handler(cleaned)
        result._validation_errors = errors
        result._raw = original
        result._rejected = sorted(rejected)
        if isinstance(original, dict):
            for collection, nested in cls.proposal_collections.items():
                new_index = 0
                for i, parent in enumerate(original.get(collection, [])):
                    if (collection, i) in rejected:
                        continue
                    result._locations[(collection, new_index)] = (collection, i)
                    if nested and isinstance(parent, dict):
                        child_index = 0
                        for j, _ in enumerate(parent.get(nested, [])):
                            if (collection, i, nested, j) in rejected:
                                continue
                            result._locations[(collection, new_index, nested, child_index)] = (collection, i, nested, j)
                            child_index += 1
                    new_index += 1
        return result

    def rejected_items(self):
        return [(path, at_path(self._raw, path)) for path in self._rejected]

    def original_index(self, *path):
        return self._locations.get(path, path)[-1]

    def merge_repair(self, repair):
        """Replace rejected slots only; a repair cannot rewrite accepted siblings."""
        merged = deepcopy(self._raw)
        for path in self._rejected:
            try:
                previous = at_path(self._raw, path)
                replacement = at_path(repair._raw, path)
                parent = at_path(self._raw, path[:2])
                new_parent = at_path(repair._raw, path[:2])
            except (KeyError, IndexError, TypeError):
                continue
            if not isinstance(previous, dict) or not isinstance(replacement, dict):
                continue
            # Keep correlation/identity fixed. Repairs may correct content/shape,
            # never move a proposal to another offered identity or fact.
            anchors = ("target_key", "entity_id", "extracted_entity_id",
                       "target_fact_id", "incoming_fact_key", "operation")
            if any(key in previous and replacement.get(key) != previous[key] for key in anchors):
                continue
            if len(path) == 4 and any(key in parent and new_parent.get(key) != parent[key]
                                      for key in ("target_key", "entity_id")):
                continue
            at_path(merged, path[:-1])[path[-1]] = deepcopy(replacement)
        result = type(self).model_validate(merged)
        result._repair_attempted = True
        return result

    def recovery_summary(self):
        return {"rejected_items": len(self._rejected), "repair_attempted": self._repair_attempted}
