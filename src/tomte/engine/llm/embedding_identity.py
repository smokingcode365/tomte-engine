"""Opaque cache identity; configuration is captured when a provider is built."""
from hashlib import sha256
import json


def openai_embedding_identity(model, dimension, endpoint):
    return sha256(json.dumps(['openai-documents-v1',str(endpoint).rstrip('/'),model.removeprefix('openai:'),dimension],
                            separators=(',',':')).encode('utf-8')).hexdigest()
