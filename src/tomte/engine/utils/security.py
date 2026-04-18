import re
import logging
from typing import Optional
from tomte.engine.exceptions import SecurityViolationError, NamespaceError
from tomte.engine.log import log


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
        self, 
        max_input_length: int = 4000, 
        enable_pii_scrubbing: bool = False
    ):
        self.max_input_length = max_input_length
        self._compiled_patterns = [re.compile(p) for p in self.INJECTION_PATTERNS]
        
        # Optional PII Scrubbing
        self.enable_pii_scrubbing = enable_pii_scrubbing
        self.analyzer = None
        self.anonymizer = None
        
        if self.enable_pii_scrubbing:
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
            log.info("PII Scrubber initialized successfully.")
            
        except ImportError:
            raise ImportError(
                "PII Scrubbing requires optional dependencies. "
                "Please install them using: pip install presidio-analyzer presidio-anonymizer"
            )
        except Exception as e:
            log.warning(f"Failed to load Presidio NLP models. Error: {e}")
            self.analyzer = None # Fallback to prevent crashing

    def validate_namespace(self, tenant_id: str, user_id: str) -> None:
        """Prevents 'Data Bleed' across tenants."""
        if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
            raise NamespaceError("A valid 'tenant_id' must be provided.")
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise NamespaceError("A valid 'user_id' must be provided.")

    def sanitize_input(self, raw_text: str) -> str:
        """Cleans input to prevent database crashes and token overflow."""
        if not raw_text:
            return ""
        clean_text = raw_text[:self.max_input_length]
        clean_text = clean_text.replace('\x00', '')
        clean_text = ''.join(c for c in clean_text if c.isprintable() or c in '\n\r\t')
        return clean_text.strip()

    def check_for_injection(self, text: str) -> None:
        """Raises an error if prompt injection is detected."""
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                raise SecurityViolationError(
                    "Potential prompt injection or malicious command detected. Request blocked."
                )

    def scrub_pii(self, text: str) -> str:
        """Masks sensitive data (e.g., replaces SSN with <US_SSN>)."""
        if not self.analyzer or not text:
            return text

        sensitive_entities = [
            "CREDIT_CARD", "CRYPTO", "EMAIL_ADDRESS", "IBAN_CODE", 
            "IP_ADDRESS", "PHONE_NUMBER", "US_SSN", "US_BANK_NUMBER"
        ]

        try:
            results = self.analyzer.analyze(text=text, entities=sensitive_entities, language="en")
            anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized_result.text
        except Exception as e:
            log.error(f"PII Scrubbing failed. Error: {e}")
            # If the scrubber crashes, drop the text entirely for absolute safety
            return "[TEXT WITHHELD DUE TO SCRUBBER FAILURE]"

    def process_incoming_request(self, tenant_id: str, user_id: str, raw_text: str) -> str:
        """
        The main entry point. Runs all security checks in order.
        """
        # 1. Check Namespace Lock
        self.validate_namespace(tenant_id, user_id)
        
        # 2. Sanitize for DB/JSON safety
        safe_text = self.sanitize_input(raw_text)
        
        # 3. Check for Prompt Injections
        self.check_for_injection(safe_text)
        
        # 4. Optional: Scrub PII
        if self.enable_pii_scrubbing:
            safe_text = self.scrub_pii(safe_text)
            
        return safe_text