from tomte.engine.database.cache_interfaces import CacheProvider
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.core.config import TomteSettings
from tomte.engine.log.interfaces import LogProvider
from tomte.engine.log import tomtelog
from tomte.engine.workers.runner import TomteWorker
from tomte.engine.clients.chat_client import TomteChatClient
from tomte.engine.clients.admin_client import TomteAdminClient

class Tomte:
    _cache: CacheProvider | None = None
    _llm: LLMProvider | None = None
    _db: DatabaseProvider = None
    _worker: TomteWorker | None = None
    @classmethod
    def init(
        cls, 
        db_provider: DatabaseProvider,
        cache_provider: CacheProvider, 
        llm_provider: LLMProvider,
        log_provider: LogProvider|None = None,
        start_daemon_worker: bool = True
    ):
        """
        Initializes the Tomte Engine with user-injected dependencies.
        """
        cls._db = db_provider
        cls._cache = cache_provider
        cls._llm = llm_provider
        if log_provider:
            tomtelog.configure_log_provider(log_provider)
        cls._worker = TomteWorker()
        if start_daemon_worker:
            cls._worker.start_in_background()

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

    @classmethod
    def get_worker(cls) -> TomteWorker:
        if not cls._worker:
            raise RuntimeError("Tomte not initialized. Call Tomte.init() first.")
        return cls._worker

    @classmethod
    def create_tomte_client(cls, tenant_id: str) -> TomteChatClient:
        return TomteChatClient(tenant_id=tenant_id)

    @classmethod
    def create_tomte_admin(cls, tenant_id: str) -> TomteAdminClient:
        return TomteAdminClient(tenant_id=tenant_id)