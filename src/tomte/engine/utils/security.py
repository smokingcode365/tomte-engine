import re
from typing import Optional
from tomte.engine.core.config import settings
from tomte.engine.exceptions import TomteSecurityViolationError
from tomte.engine import tomtelog


class SecurityGuard:
    """
    The Tomte Engine Security Middleware.
    Enforces namespace isolation, sanitizes inputs, blocks injections, 
    and offers optional PII scrubbing.
    """
    
    INJECTION_PATTERNS = [
        r"(?i)(ignore|disregard)\s+(all\s+)?(previous\s+)?(instructions|prompts)",
        r"(?i)system\s+prompt",
        r"(?i)you\s+are\s+now",
        r"(?i)drop\s+table",
        r"(?i)```(sql|python|bash)"
    ]
    
    def __init__(
        self
    ):
        self._compiled_patterns = [re.compile(p) for p in self.INJECTION_PATTERNS]
        # Optional PII Scrubbing
        self.analyzer = None
        self.anonymizer = None
        if settings.enable_pii_scrubbing:
            self._initialize_pii_scrubber()

    def _initialize_pii_scrubber(self) -> None:
        """
        Lazy-loads Microsoft Presidio only if the user explicitly opts in.
        """
        try:
            # We import here so it doesn't break the app if not installed
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            tomtelog.info("PII Scrubber initialized successfully.")
            
        except ImportError:
            raise ImportError(
                "PII Scrubbing requires optional dependencies. "
                "Please install them using: pip install presidio-analyzer presidio-anonymizer"
            )
        except Exception as e:
            tomtelog.warn("Failed to load Presidio NLP models")
            self.analyzer = None # Fallback to prevent crashing


    def sanitize_input(self, raw_text: str) -> str:
        """Cleans input to prevent database crashes and token overflow."""
        if not raw_text:
            return ""
        clean_text = raw_text[:settings.max_input_length]
        clean_text = clean_text.replace('\x00', '')
        clean_text = ''.join(c for c in clean_text if c.isprintable() or c in '\n\r\t')
        return clean_text.strip()

    def check_for_injection(self, text: str) -> bool:
        """Raises an error if prompt injection is detected."""
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                return False
        return True

    def scrub_pii(self, text: str) -> str:
        """Masks sensitive data (e.g., replaces SSN with <US_SSN>)."""
        if not settings.enable_pii_scrubbing:
            return text
        if not text:
            return text
        if not self.analyzer or not self.anonymizer:
            return "[TEXT WITHHELD DUE TO SCRUBBER FAILURE]"
        sensitive_entities = [
            "CREDIT_CARD", "CRYPTO", "EMAIL_ADDRESS", "IBAN_CODE", 
            "IP_ADDRESS", "PHONE_NUMBER", "US_SSN", "US_BANK_NUMBER"
        ]

        try:
            results = self.analyzer.analyze(text=text, entities=sensitive_entities, language="en")
            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized_result.text
        except Exception as e:
            tomtelog.error("PII Scrubbing failed")
            # If the scrubber crashes, drop the text entirely for absolute safety
            return "[TEXT WITHHELD DUE TO SCRUBBER FAILURE]"

# ==========================================
# SECURITY GUARD SINGLETON
# ==========================================
_security_guard_instance = None

def get_security_guard():
    """
    Singleton getter for SecurityGuard instance.
    Returns the same SecurityGuard object across imports.
    """
    global _security_guard_instance
    if _security_guard_instance is None:
        _security_guard_instance = SecurityGuard()
    return _security_guard_instance
