from typing import Optional
from tomte.engine.log.interfaces import LogProvider
from tomte.engine.providers.log_provider import DefaultLogProvider

# Internal state: the active LogProvider implementation
_provider: LogProvider = DefaultLogProvider()


def info(msg: str) -> None:
    _provider.info(msg)


def warn(msg: str) -> None:
    _provider.warn(msg)


def warning(msg: str) -> None:
    """Alias for :func:`warn` (stdlib ``logging``-style name)."""
    _provider.warn(msg)


def debug(msg: str) -> None:
    _provider.debug(msg)


def error(msg: str, exc: Optional[Exception] = None) -> None:
    _provider.error(msg, exc)


def configure_log_provider(custom_provider: LogProvider) -> None:
    """
    Call once at application startup to replace the default logger
    (e.g. with loguru or structured logging).
    """
    global _provider
    _provider = custom_provider
