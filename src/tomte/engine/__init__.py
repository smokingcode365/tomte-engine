from __future__ import annotations
from contextlib import contextmanager
from threading import RLock
from functools import wraps

from tomte.engine.database.cache_interfaces import CacheProvider
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.core.config import TomteSettings
from tomte.engine.log.interfaces import LogProvider
from tomte.engine.log import tomtelog

def serialized(method):
    @wraps(method)
    def call(cls, *args, **kwargs):
        with cls._init_lock:
            return method(cls, *args, **kwargs)
    return call


class Tomte:
    _init_lock = RLock()
    _operation_factory = None
    _usage = None
    _active_operations = 0
    _closing = False
    _init_options = None
    _cache: CacheProvider | None = None
    _llm: LLMProvider | None = None
    _db: DatabaseProvider = None
    _worker: TomteWorker | None = None
    @classmethod
    @serialized
    def init(
        cls, 
        db_provider: DatabaseProvider,
        cache_provider: CacheProvider, 
        llm_provider: LLMProvider,
        log_provider: LogProvider|None = None,
        start_daemon_worker: bool = True,
        *, max_workers: int = 1, operation_factory=None, usage_listener=None
    ):
        """
        Initializes the Tomte Engine with user-injected dependencies.
        """
        options = (max_workers, operation_factory, usage_listener, log_provider, start_daemon_worker)
        if cls._worker is not None:
            if (not cls._closing and cls._db is db_provider and cls._cache is cache_provider
                    and cls._llm is llm_provider and cls._init_options == options):
                return
            raise RuntimeError('Tomte already initialized; shutdown before replacing providers')
        if type(max_workers) is not int or not 1 <= max_workers <= 32:
            raise ValueError('max_workers must be between 1 and 32')
        if operation_factory is not None and not callable(operation_factory):
            raise ValueError('operation_factory must be callable')
        if usage_listener is not None and not callable(usage_listener):
            raise ValueError('usage_listener must be callable')
        if start_daemon_worker and operation_factory is None:
            from tomte.engine.providers.postgres_provider import PostgresProvider
            from tomte.engine.providers.openai_provider import OpenAILLMProvider
            if type(db_provider) is not PostgresProvider or type(llm_provider) is not OpenAILLMProvider:
                raise ValueError('Managed custom providers require operation_factory')
        cls._closing = False
        cls._init_options = options
        cls._db = db_provider
        cls._cache = cache_provider
        cls._llm = llm_provider
        if log_provider:
            tomtelog.configure_log_provider(log_provider)
        from tomte.engine.usage import UsageReporter
        cls._operation_factory = operation_factory
        cls._usage = UsageReporter(usage_listener)
        cls._worker = TomteWorker(max_workers=max_workers, usage=cls._usage)
        if start_daemon_worker:
            cls._worker.start_in_background()

    @classmethod
    @contextmanager
    def operation_context(cls, control=None, *, cache_only=False):
        from tomte.engine.runtime import owned_operation
        with cls._init_lock:
            if cls._closing:
                raise RuntimeError('Tomte is shutting down')
            factory, db, llm = cls._operation_factory, cls.get_db(), cls.get_llm()
            cls._active_operations += 1
        try:
            from contextlib import nullcontext
            context = nullcontext((None, None)) if cache_only else (
                factory(control) if factory else owned_operation(db, llm, control))
            with context as resources:
                yield resources
        finally:
            with cls._init_lock:
                cls._active_operations -= 1

    @classmethod
    def usage(cls, tenant_id=None, user_id=None):
        pending = None
        if tenant_id is not None and user_id is not None:
            try:
                pending = cls.get_cache().pending_input_chars(tenant_id, user_id)
            except Exception:
                pass
        return cls._usage.query(tenant_id, user_id, pending) if cls._usage else None

    @classmethod
    def shutdown(cls, timeout=5):
        from time import monotonic
        deadline = monotonic() + max(0, timeout)
        with cls._init_lock:
            cls._closing = True
            worker, usage = cls._worker, cls._usage
        if worker and not worker.stop(max(0, deadline-monotonic())):
            return False
        if cls._active_operations:
            return False
        if usage and not usage.close(min(max(0, deadline-monotonic()), 1)):
            return False
        with cls._init_lock:
            if cls._worker is not worker:
                return cls._worker is None
            cls._worker = cls._db = cls._cache = cls._llm = None
            cls._operation_factory = cls._usage = None
            return True

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
    def _validate_init(cls):
        if not cls._db or not cls._cache or not cls._llm:
            raise RuntimeError("Tomte not initialized. Call Tomte.init() first.")

    # ==========================================
    @classmethod
    def create_tomte_client(cls, tenant_id: str) -> TomteChatClient:
        cls._validate_init()
        return TomteChatClient(tenant_id=tenant_id)

    @classmethod
    def create_tomte_admin(cls, tenant_id: str) -> TomteAdminClient:
        cls._validate_init()
        return TomteAdminClient(tenant_id=tenant_id)


# These modules import Tomte, so load them after the class is defined.
from tomte.engine.workers.runner import TomteWorker
from tomte.engine.clients.chat_client import TomteChatClient
from tomte.engine.clients.admin_client import TomteAdminClient
