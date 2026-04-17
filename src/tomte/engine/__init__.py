from tomte.engine.database.interfaces import CacheProvider
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.core.config import TomteSettings

class Tomte:
    _cache: CacheProvider | None = None
    _llm: LLMProvider | None = None
    _pg_url: str | None = None
    
    @classmethod
    def init(
        cls, 
        postgres_url: str, 
        cache_provider: CacheProvider, 
        llm_provider: LLMProvider
    ):
        """
        Initializes the Tomte Engine with user-injected dependencies.
        """
        cls._pg_url = postgres_url
        cls._cache = cache_provider
        cls._llm = llm_provider
        
        # We can also initialize our internal settings here
        global settings
        settings = TomteSettings(postgres_url=postgres_url)

    @classmethod
    def get_cache(cls) -> CacheProvider:
        if not cls._cache:
            raise RuntimeError("Tomte not initialized. Call Tomte.init() first.")
        return cls._cache

    @classmethod
    def get_llm(cls) -> LLMProvider:
        if not cls._llm:
            raise RuntimeError("Tomte not initialized.")
        return cls._llm