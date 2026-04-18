from tomte.engine.database.interfaces import CacheProvider, DatabaseProvider
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.core.config import TomteSettings

class Tomte:
    _cache: CacheProvider | None = None
    _llm: LLMProvider | None = None
    _db: DatabaseProvider = None
    @classmethod
    def init(
        cls, 
        db_provider: DatabaseProvider,
        cache_provider: CacheProvider, 
        llm_provider: LLMProvider
    ):
        """
        Initializes the Tomte Engine with user-injected dependencies.
        """
        cls._db = db_provider
        cls._cache = cache_provider
        cls._llm = llm_provider
        
    @classmethod
    def get_db(cls) -> DatabaseProvider:
        if not cls._db:
            raise RuntimeError("Tomte not initialized. Call Tomte.init() first.")
        return cls._db

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