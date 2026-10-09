from typing import Optional, Tuple

from tomte.engine.core.config import settings


def generate_common_key(tenant_id: str, session_id: str, user_id: Optional[str] = None) -> str:
    if not user_id:
        if not session_id:
            raise ValueError("Guests must have a session_id to maintain chat history.")
        return f"Tomte:chat:guest:{tenant_id}:{session_id}"
        
    if settings.isolate_chat_sessions:
        return f"Tomte:chat:user:{tenant_id}:{user_id}:{session_id}"
    else:
        return f"Tomte:chat:user:{tenant_id}:{user_id}"
        
def generate_consolidation_key(tenant_id: str, user_id: str, session_id: str) -> str:
    return f"Tomte:consolidation:queue:{tenant_id}:{user_id}:{session_id}"

def generate_consolidation_batch_key(tenant_id: str, user_id: str, session_id: str) -> str:
    """Durable Redis snapshot awaiting PostgreSQL commit and acknowledgment."""
    return generate_consolidation_key(tenant_id, user_id, session_id) + ":processing"


def generate_session_ref(tenant_id: str, session_id: str, user_id:str) -> str:
    return f"{tenant_id}:{user_id}:{session_id}"

def generate_persona_key(tenant_id: str, user_id: str) -> str:
    return f"Tomte:persona:{tenant_id}:{user_id}"

def parse_session_ref(session_ref: str) -> Tuple[str, str, str]:
    return session_ref.split(":")

def get_section_key_active_sessions() -> str:
    return "Tomte:consolidation:active_sessions"

def validate_id(id: str, can_be_empty: bool = False) -> bool:
    """
    Validates an ID according to the following rules:
    - Less than 100 characters
    - Does not contain ':' or space
    - Does not contain non-printable (invisible) characters
    - If can_be_empty is False, the ID cannot be empty
    """
    if id is None:
        if not can_be_empty:
            return False
        return True
    if not isinstance(id, str):
        return False
    if len(id) > 100:
        return False
    if len(id) == 0:
        return False
    if ':' in id or ' ' in id or not all(c.isprintable() and c not in '\n\r\t' for c in id):
        return False
    return True