"""Execute one focused memory request with vector search and batched fact reading."""
import json

from tomte.engine.core.answer_facts import checked_packets, packet_key
from tomte.engine.core.answering import validate_memory_owner_name
from tomte.engine.core.config import settings
from tomte.engine.core.memory import public_entity
from tomte.engine.core.unresolved import pending_records
from tomte.engine.exceptions import TomteInvalidInputError
from tomte.engine.workflows.fact_reading import BatchedFactReader, MAX_FACT_CANDIDATES
from tomte.engine.workflows.memory_search import read_memory_views
from tomte.engine.workflows.pending_search import PendingMemorySearch
from tomte.engine.workflows.retrieval_evidence import EvidenceBuffer, RetrievalResult
from tomte.engine.workflows.search_request import prepare_embedding_query


class MemoryRetrievalWorkflow:
    def __init__(self, tenant, user, db, llm, cache, control, *, memory_owner_name=None):
        self.tenant, self.user = tenant, user
        self.db, self.llm, self.cache, self.control = db, llm, cache, control
        self.memory_owner_name = validate_memory_owner_name(memory_owner_name)
        self.pending = None
        self.embedding_queries = {}
        self.fact_search = (getattr(db, 'supports_answer_fact_search', False) is True
                            and settings.stored_retrieval_mode == 'vector')
        self.reader = BatchedFactReader(llm, control, memory_owner_name=self.memory_owner_name)

    def _read_facts(self, information_needed, vector, result):
        config = settings.answer_retrieval
        limit = min(config.fact_limit, MAX_FACT_CANDIDATES)
        self.control.check()
        # Empty paths requests one broad ranking using existing plain-value vectors.
        rows = self.db.get_answer_facts(self.tenant, self.user, vector, [], limit, config.lane_bytes)
        self.control.check()
        offered = {}
        for packet in checked_packets(rows, self.tenant, self.user):
            offered.setdefault(packet_key(packet), packet)
        packets = list(offered.values())[:limit]
        result.observed_packets = packets
        result.coverage.append({'lane': 'broad', 'total_facts': rows['total_facts'],
            'offered_facts': len(packets), 'is_partial': rows['total_facts'] > len(packets),
            'oversized_facts': rows['oversized_facts']})
        result.packets, result.possible_packets, reading_coverage = self.reader.read(information_needed, packets)
        result.coverage.extend(reading_coverage)
        result.unresolved = [public_entity(r['payload']) for r in pending_records(
            self.db, self.tenant, self.user, vector, settings.unresolved_return_limit)]

    def retrieve_once(self, information_needed):
        """Return this request's evidence only; never plan a follow-up or an answer."""
        if (not isinstance(information_needed, str) or not information_needed.strip()
                or '\x00' in information_needed or len(information_needed) > settings.max_input_length):
            raise TomteInvalidInputError('Invalid memory request')
        request = information_needed.strip()
        self.control.check()
        if request not in self.embedding_queries:
            self.embedding_queries[request] = prepare_embedding_query(request, self.llm, self.control)
        if self.pending is None:
            # Capture before durable reads, then reuse across answer rounds.
            self.pending = PendingMemorySearch(self.tenant, self.user, self.db, self.cache, self.control)
        result = RetrievalResult(snapshot_at=self.pending.snapshot_at)
        vector = self.llm.embed_text(self.embedding_queries[request])
        self.control.check()
        result.pending = self.pending.search(vector, self.llm)
        self.control.check()
        if self.fact_search:
            self._read_facts(request, vector, result)
        else:
            result.views, result.unresolved = read_memory_views(
                self.tenant, self.user, request, vector, settings.semantic_retrieval_limit, self.db)
        self.control.check()
        return result

    def run(self, information_needed):
        """Public retrieval renders one pass, with no accumulated answer state."""
        evidence = EvidenceBuffer()
        evidence.merge(self.retrieve_once(information_needed))
        memory = evidence.render()
        if self.memory_owner_name is not None:
            memory = 'Memory owner: ' + json.dumps(self.memory_owner_name, ensure_ascii=False) + '\n' + memory
        return memory
