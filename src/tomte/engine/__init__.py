from tomte.engine.database.cache_interfaces import CacheProvider
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.core.config import TomteSettings
from tomte.engine.log.interfaces import LogProvider
from tomte.engine.log.log import configure_log_provider

class Tomte:
    _cache: CacheProvider | None = None
    _llm: LLMProvider | None = None
    _db: DatabaseProvider = None
    @classmethod
    def init(
        cls, 
        db_provider: DatabaseProvider,
        cache_provider: CacheProvider, 
        llm_provider: LLMProvider,
        log_provider: LogProvider|None = None,
        isolate_chat_sessions: bool = False
    ):
        """
        Initializes the Tomte Engine with user-injected dependencies.
        """
        cls._db = db_provider
        cls._cache = cache_provider
        cls._llm = llm_provider
        if log_provider:
            configure_log_provider(log_provider)
        TomteSettings.isolate_chat_sessions = isolate_chat_sessions
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