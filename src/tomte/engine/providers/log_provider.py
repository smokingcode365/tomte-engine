import logging
from typing import Optional
from tomte.engine.log.interfaces import LogProvider

class DefaultLogProvider(LogProvider):
    """
    A standard implementation of the LogProvider interface using 
    the Python built-in logging module.
    """
    def __init__(self, name: str = "tomte.engine", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        
        # We set the level on the logger itself, but we avoid adding handlers
        # here to prevent "double-logging" in host applications that have 
        # their own logging configuration.
        self.logger.setLevel(level)

    def info(self, message: str):
        """Logs an informational message."""
        self.logger.info(message)

    def warn(self, message: str):
        """Logs a warning message."""
        # mapping 'warn' interface to 'warning' implementation
        self.logger.warning(message)

    def error(self, message: str, exc: Optional[Exception] = None):
        """
        Logs an error message. 
        If an exception is provided, it captures the stack trace.
        """
        if exc:
            # exc_info=exc ensures the specific exception context is preserved
            self.logger.error(message, exc_info=exc)
        else:
            self.logger.error(message)

    def debug(self, message: str):
        """Logs a debug-level message."""
        self.logger.debug(message)