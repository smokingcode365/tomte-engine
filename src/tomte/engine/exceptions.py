"""
Public Exceptions for the Tomte Engine.
"""

class TomteError(Exception):
    """Base exception for all Tomte Engine errors."""
    pass

class TomteConfigurationError(TomteError):
    """Raised when the engine is initialized with missing or invalid credentials."""
    pass

class TomteLockError(TomteError):
    """Raised when a background worker fails to acquire the Redis tenant/user Mutex lock."""
    pass

class SchemaValidationError(TomteError):
    """Raised when the LLM returns data that violates the Bounded Hybrid architecture."""
    pass

class SecurityViolationError(Exception):
    """Raised when an active prompt injection or malicious input is detected."""
    pass

class NamespaceError(Exception):
    """Raised when tenant_id or user_id are missing or malformed."""
    pass

class DatabaseConnectionError(Exception):
    pass