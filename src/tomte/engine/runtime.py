"""Owned resources for one synchronous foreground call or background attempt."""
import asyncio
from contextlib import contextmanager
from tomte.engine.workers.control import WorkControl


class ControlledLLM:
    def __init__(self, provider, control):
        self.provider, self.control = provider, control

    def __getattr__(self, name):
        if name not in {'embed_text', 'embed_batch', 'is_junk', 'parse_structured'}:
            return getattr(self.provider, name)
        def invoke(*args, **kwargs):
            self.control.check()
            result = getattr(self.provider, name)(*args, **kwargs)
            self.control.check()
            return result
        return invoke


@contextmanager
def owned_operation(db_template, llm_template, control=None):
    """The default native OpenAI path. Custom providers supply a context factory.

    Existing standalone provider objects remain supported by direct workflows.
    They are never silently shared by the managed concurrent runtime.
    """
    from tomte.engine.providers.postgres_provider import PostgresProvider
    from tomte.engine.providers.openai_provider import OpenAILLMProvider
    if type(db_template) is not PostgresProvider or type(llm_template) is not OpenAILLMProvider:
        raise RuntimeError('Custom providers require an operation_factory for managed execution')
    db, lazy = None, LazyOpenAI()
    try:
        db = db_template.fork()
        yield db, ControlledLLM(lazy, control or WorkControl())
    finally:
        try:
            lazy.close()
        finally:
            if db is not None:
                db.close()


class LazyOpenAI:
    """One operation's model resources, allocated only on first model access."""
    def __init__(self):
        self.instance = self.client = self.loop = self.previous = None

    def __getattr__(self, name):
        if self.instance is None:
            try:
                self._initialize()
            except BaseException:
                try:
                    self.close()
                except Exception:
                    pass
                raise
        return getattr(self.instance, name)

    def _initialize(self):
        from openai import AsyncOpenAI
        from pydantic_ai import Embedder
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
        from tomte.engine.providers.openai_provider import OpenAILLMProvider
        from tomte.engine.llm.structured import make_agents
        from tomte.engine.core.config import settings
        try:
            self.previous = asyncio.get_event_loop()
        except RuntimeError:
            self.previous = None
        if self.previous is not None and self.previous.is_running():
            raise RuntimeError('Use the synchronous client from a non-event-loop thread')
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # Keep existing environment-based credential precedence.
        self.client = AsyncOpenAI(timeout=30, max_retries=0)
        client = self.client
        provider = OpenAIProvider(openai_client=client)
        llm = object.__new__(OpenAILLMProvider)
        llm.strong_model = OpenAIResponsesModel(settings.openai.strong_model, provider=provider)
        llm.cheap_model = OpenAIResponsesModel(settings.openai.cheap_model, provider=provider)
        llm.embedder = Embedder(OpenAIEmbeddingModel(settings.openai.embed_model.removeprefix('openai:'), provider=provider))
        llm.extractor_agent, llm.junk_agent = make_agents()
        llm.agent_retries = 0
        self.instance = llm

    def close(self):
        if self.loop is None:
            return
        try:
            if self.client is not None:
                self.loop.run_until_complete(self.client.close())
        finally:
            self.loop.close()
            asyncio.set_event_loop(self.previous)
            self.instance = self.client = self.loop = self.previous = None
