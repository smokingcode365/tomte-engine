"""Separate model-interpreted search wording from the full filtering request."""
from pydantic import field_validator

from tomte.engine.core.config import settings
from tomte.engine.core.memory import MemoryModel, nonempty_text
from tomte.engine.exceptions import TomteInvalidInputError
from tomte.engine.llm.calls import call_model


SEARCH_EMBEDDING_PROMPT = """Return the embedding_text for this memory information request.
Remove search exclusions and the excluded items, not just the negation word:
"Outdoor activities done or considered besides hiking" becomes
"Outdoor activities done or considered". Keep the remaining wording, subject,
requested detail and time. Do not answer, add examples, synonyms or guessed facts.
If there is no exclusion, copy the request unchanged. A negative fact being sought
is not an exclusion: keep "medicines I cannot take" and "why I stopped hiking".
This applies by meaning in any language. The full original request is retained
separately for filtering; return only the positive search text, no explanation.
"""


class SearchEmbeddingQuery(MemoryModel):
    embedding_text: str

    _text = field_validator('embedding_text')(nonempty_text)


def prepare_embedding_query(information_needed, llm, control):
    query = call_model(llm, control, SEARCH_EMBEDDING_PROMPT,
                       {'information_needed': information_needed}, SearchEmbeddingQuery).embedding_text.strip()
    if len(query) > settings.max_input_length:
        raise TomteInvalidInputError('Embedding request exceeds the input limit')
    return query
