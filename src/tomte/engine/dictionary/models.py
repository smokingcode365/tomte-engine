from pydantic import BaseModel
from typing import List

class DictionaryItem(BaseModel):
    keyword: str
    keyword_type: str
    hits: int
    keyword_context: str = ""

class BootstrapDictionary(BaseModel):
    entity_keywords: List[str]
    attribute_keywords: List[str]
