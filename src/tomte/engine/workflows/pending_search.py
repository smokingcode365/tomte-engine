"""Read-only pending evidence with a disposable, ownership-checked vector cache."""
from copy import deepcopy
import math
from time import monotonic, sleep
from uuid import uuid4

from tomte.engine.core.config import settings
from tomte.engine.core.pending_windows import plan_windows, public_windows, stamp, window_key
from tomte.engine.exceptions import TomteSecurityViolationError, TomteSessionStateUnavailable
from tomte.engine.llm.embedder import embed_texts, EMBED_BATCH_SIZE
from tomte.engine.utils.keys import validate_id
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


def _list(value):
    # Redis cjson encodes empty Lua arrays as {}.
    if value == {}:
        return []
    if not isinstance(value, list):
        raise ValueError('Invalid pending snapshot list')
    return value


def _valid_vector(vector, dimension):
    return (isinstance(vector, list) and len(vector) == dimension
            and all(type(v) in (int,float) and math.isfinite(v) for v in vector) and any(vector))


class PendingMemorySearch:
    """Capture once, then reuse the same source snapshot/vectors for more queries.

    Construct BEFORE durable retrieval to avoid the queue-to-memory handoff gap.
    This does not claim a distributed or fully point-in-time PostgreSQL snapshot.
    """
    def __init__(self, tenant, user, db, cache, control=None):
        self.tenant, self.user, self.cache, self.control = tenant, user, cache, control
        self.cfg = settings.pending_search.model_copy(deep=True)
        self.reasons, self.windows, self.vectors = set(), [], {}
        self.messages, self.observed, self.snapshot_at = 0, 0, None
        self.embedding_bytes = 0
        self.owner = uuid4().hex
        try:
            snapshot = cache.get_pending_search_snapshot(tenant, user)
        except (TomteSessionStateUnavailable, NotImplementedError):
            self.reasons.add('pending_snapshot_unavailable')
            return
        self.snapshot_at = stamp(snapshot['snapshot_at'])
        self.observed = snapshot['observed_messages']
        if not snapshot['complete']:
            self.reasons.add('snapshot_limit_or_invalid_input')
        anchors = _list(snapshot.get('anchors', []))
        seen_sessions = set()
        for batch in _list(snapshot['sessions']):
            sid = batch['session_id']
            if not validate_id(sid) or sid in seen_sessions:
                raise ValueError('Invalid pending snapshot session')
            seen_sessions.add(sid)
            processing, queued, identity = (_list(batch.get(k, [])) for k in ('processing','queued','identity'))
            for item in [*identity, *processing, *queued]:
                if (item.get('tenant_id'),item.get('user_id'),item.get('session_id')) != (tenant,user,sid):
                    raise TomteSecurityViolationError('Pending search namespace mismatch')
            if identity and db.is_consolidation_processed(tenant, user,
                    ConsolidationWorkflow._batch_id(tenant,user,identity)):
                processing = []
            messages, ids = [], set()
            for status, items in (('processing', processing), ('queued', queued)):
                for source in items:
                    item = deepcopy(source)
                    mid, text, timestamp = item.get('item_id'), item.get('content'), item.get('timestamp')
                    if (not isinstance(mid,str) or not mid or mid in ids or not isinstance(text,str)
                            or item.get('role') not in ('user','assistant')
                            or type(timestamp) not in (int,float) or not math.isfinite(timestamp)):
                        raise ValueError('Invalid pending search message')
                    text.encode('utf-8'); stamp(timestamp)
                    ids.add(mid)
                    item['status'] = status
                    refs = _list(item.get('recent_context', []))
                    for ref in refs:
                        if ref.get('role') not in ('user','assistant') or not isinstance(ref.get('content'),str):
                            raise ValueError('Invalid pending reference context')
                    item['recent_context'] = [{'role':r['role'],'content':r['content']} for r in refs[-4:]]
                    messages.append(item)
            self.messages += len(messages)
            own_anchors = [a for a in anchors if a.get('session_id') == sid]
            for window in plan_windows(messages, own_anchors, self.cfg.window_bytes):
                window.update(session_id=sid, generation=batch['generation'])
                self.windows.append(window)

    def _check(self):
        if self.control is not None:
            self.control.check()

    def _release(self, window):
        try:
            self.cache.release_pending_window(self.tenant,self.user,window['session_id'],window['key'],self.owner)
        except (TomteSessionStateUnavailable, NotImplementedError):
            pass  # A failed optional release is bounded by the claim lease.

    def _claim(self, window, use_cache):
        if not use_cache:
            return {'state':'local'}
        try:
            return self.cache.claim_pending_window(self.tenant,self.user,window['session_id'],
                window['generation'],window['key'],
                {'session_id':window['session_id'],'parts':window['parts'],'closed':window['closed']},self.owner)
        except (TomteSessionStateUnavailable, NotImplementedError):
            return {'state':'local'}

    def _embed(self, llm, windows, owned):
        if not windows:
            return
        self._check()
        vectors = embed_texts(llm, [w['text'] for w in windows])
        for window, vector in zip(windows,vectors):
            self.vectors[window['key']] = vector
            if window['key'] in owned:
                try:
                    self.cache.publish_pending_window(self.tenant,self.user,window['session_id'],
                        window['generation'],window['key'],self.owner,vector)
                except (TomteSessionStateUnavailable, NotImplementedError):
                    pass  # This call already has the result; persistence is optional.

    def search(self, query_vector, llm):
        dimension = len(query_vector)
        if not _valid_vector(query_vector, dimension):
            raise ValueError('Invalid pending query embedding')
        get_identity = getattr(llm, 'get_embedding_cache_identity', None)
        identity = get_identity() if callable(get_identity) else None
        use_cache = isinstance(identity,str) and bool(identity)
        identity = [identity if use_cache else self.owner, dimension]
        owned, work, busy = {}, [], []
        current_reasons = set(self.reasons)
        # Prefer recent windows when the cold-embedding allowance is exhausted.
        ordered = sorted(self.windows, key=lambda w:w['units'][-1]['timestamp'], reverse=True)
        try:
            for window in ordered:
                self._check()
                window['key'] = window_key(self.tenant,self.user,window['session_id'],identity,window)
                if window['key'] in self.vectors:
                    continue
                claim = self._claim(window,use_cache)
                if claim['state'] == 'ready':
                    if not _valid_vector(claim['vector'], dimension):
                        raise ValueError('Invalid cached pending embedding')
                    self.vectors[window['key']] = claim['vector']
                elif claim['state'] == 'busy':
                    busy.append(window)
                elif claim['state'] in ('owned','local','full','retired'):
                    if claim['state'] == 'owned': owned[window['key']] = window
                    size = len(window['text'].encode('utf-8'))
                    if self.embedding_bytes + size > self.cfg.max_embedding_bytes:
                        current_reasons.add('embedding_budget')
                        continue
                    self.embedding_bytes += size
                    work.append(window)
                    if len(work) == EMBED_BATCH_SIZE:
                        self._embed(llm,work,owned); work=[]
                else:
                    raise ValueError('Invalid pending cache claim')
            self._embed(llm,work,owned)
            deadline = monotonic() + self.cfg.wait_seconds
            while busy and monotonic() < deadline:
                self._check()
                remaining = []
                for window in busy:
                    claim = self._claim(window,use_cache)
                    if claim['state'] == 'ready':
                        if not _valid_vector(claim['vector'],dimension):
                            raise ValueError('Invalid cached pending embedding')
                        self.vectors[window['key']] = claim['vector']
                    elif claim['state'] == 'busy':
                        remaining.append(window)
                    elif claim['state'] in ('owned','local','full','retired'):
                        if claim['state'] == 'owned': owned[window['key']] = window
                        size = len(window['text'].encode('utf-8'))
                        if self.embedding_bytes + size > self.cfg.max_embedding_bytes:
                            current_reasons.add('embedding_budget')
                            continue
                        self.embedding_bytes += size
                        self._embed(llm,[window],owned)
                    else:
                        raise ValueError('Invalid pending cache claim')
                busy = remaining
                if busy: sleep(min(.02,max(0,deadline-monotonic())))
            if busy: current_reasons.add('embedding_in_progress')
        finally:
            for window in owned.values(): self._release(window)
        norm = math.hypot(*query_vector)
        scored, indexed = [], set()
        for window in self.windows:
            vector = self.vectors.get(window['key'])
            if vector is None: continue
            vector_norm = math.hypot(*vector)
            score = sum((a/norm)*(b/vector_norm) for a,b in zip(query_vector,vector))
            scored.append((score,window))
            indexed.update((window['session_id'],*part) for part in map(tuple,window['parts']))
        ranked = [w for _,w in sorted(scored,key=lambda row:(-row[0],row[1]['key']))]
        selected = public_windows(ranked,self.cfg.return_windows,self.cfg.return_bytes)
        if ranked and not selected: current_reasons.add('return_budget')
        required = {(w['session_id'],*part) for w in self.windows for part in map(tuple,w['parts'])}
        return {'pending_message_windows':selected, 'pending_search':{
            'complete':not current_reasons and indexed==required,
            'reasons':sorted(current_reasons), 'snapshot_at':self.snapshot_at,
            'observed_queue_messages':self.observed, 'messages_considered':self.messages,
            'windows_searched':len(scored), 'windows_returned':len(selected),
        }}
