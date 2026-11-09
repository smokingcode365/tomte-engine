"""Disposable pending-search cache. Authoritative queues never live in these keys."""
import json
from uuid import uuid4
import redis
from tomte.engine.core.config import settings
from tomte.engine.exceptions import TomteSessionStateUnavailable
from tomte.engine.utils.keys import validate_id, get_section_key_active_sessions


def keys(tenant, user, session):
    if not all(validate_id(v) for v in (tenant, user, session)):
        raise ValueError('Invalid pending cache namespace')
    prefix = f'Tomte:pending-search:{tenant}:{user}'
    return prefix+':windows', prefix+':vectors', prefix+':epoch:'+session


# Included in BOTH queue acknowledgement scripts. Completion deletes the epoch
# before any late publisher can recreate a retired window. Malformed disposable
# cache entries are removed; they cannot prevent authoritative queue completion.
RETIRE_LUA = """
local function retire_windows(catalog, vectors, epoch, session, raw_items)
    redis.call('del', epoch)
    local kind = redis.call('type', catalog).ok
    if kind ~= 'none' and kind ~= 'hash' then
        redis.call('del', catalog, vectors)
        return
    end
    local ids = {}
    local invalid = false
    for _, raw in ipairs(raw_items) do
        local ok, item = pcall(cjson.decode, raw)
        if ok and type(item) == 'table' and type(item.item_id) == 'string' then
            ids[item.item_id] = true
        else invalid = true end
    end
    local rows = redis.call('hgetall', catalog)
    for i = 1, #rows, 2 do
        local ok, entry = pcall(cjson.decode, rows[i+1])
        local remove = not ok or type(entry) ~= 'table'
        if not remove and entry.session_id == session then
            remove = invalid or type(entry.parts) ~= 'table'
            if not remove then
                for _, part in ipairs(entry.parts) do
                    if type(part) ~= 'table' or ids[part[1]] then remove = true; break end
                end
            end
        end
        if remove then
            redis.call('hdel', catalog, rows[i])
            if redis.call('type', vectors).ok == 'hash' then redis.call('hdel', vectors, rows[i]) end
        end
    end
end
"""

_CACHE = """
local clock = redis.call('time')
local now = tonumber(clock[1]) + tonumber(clock[2])/1000000
for i = 1, 2 do
    local kind = redis.call('type', KEYS[i]).ok
    if kind ~= 'none' and kind ~= 'hash' then redis.call('del', KEYS[1], KEYS[2]) end
end
local rows = redis.call('hgetall', KEYS[1])
local count, bytes = 0, 0
local oldest_key, oldest_time = nil, math.huge
for i = 1, #rows, 2 do
    local ok, entry = pcall(cjson.decode, rows[i+1])
    local valid = ok and type(entry) == 'table'
    if valid then
        valid = type(entry.expires_at) == 'number' and entry.expires_at > now
            and type(entry.nbytes) == 'number' and entry.nbytes >= 0
            and type(entry.used_at) == 'number' and type(entry.ready) == 'boolean'
            and type(entry.closed) == 'boolean' and type(entry.session_id) == 'string'
            and type(entry.parts) == 'table' and #entry.parts > 0 and #entry.parts <= 6
        if valid and not entry.ready then
            valid = type(entry.lease_until) == 'number' and type(entry.owner) == 'string'
                and type(entry.generation) == 'string'
        end
        if valid then
            for _, part in ipairs(entry.parts) do
                if type(part) ~= 'table' or type(part[1]) ~= 'string'
                    or type(part[2]) ~= 'number' or type(part[3]) ~= 'number' then
                    valid = false; break
                end
            end
        end
    end
    if not valid then
        redis.call('hdel', KEYS[1], rows[i])
        redis.call('hdel', KEYS[2], rows[i])
    else
        count = count + 1
        bytes = bytes + (entry.nbytes or 0)
        if entry.ready and (entry.used_at or 0) < oldest_time then
            oldest_key, oldest_time = rows[i], entry.used_at or 0
        end
    end
end
"""


class PendingWindowCache:
    def __init__(self, client):
        self.client = client

    def _eval(self, script, key_list, *args):
        try:
            return self.client.eval(script, len(key_list), *key_list, *args)
        except (redis.ConnectionError, redis.TimeoutError):
            raise TomteSessionStateUnavailable('Pending search cache unavailable') from None

    def snapshot(self, tenant, user):
        """Atomic Redis snapshot, bounded by count and normalized input bytes."""
        catalog, vectors, _ = keys(tenant, user, '_snapshot')
        cfg = settings.pending_search
        script = _CACHE + """
        local sessions, total, used, taken = {}, 0, 0, 0
        local complete = true
        local prefix = ARGV[1]
        local refs = redis.call('zrevrange', KEYS[3], 0, -1)
        local function take(list, n)
            local result = {}
            for i = n-1, 0, -1 do
                if taken >= tonumber(ARGV[2]) then complete = false; break end
                local raw = redis.call('lindex', list, i)
                local ok, item = pcall(cjson.decode, raw or '')
                if not ok or type(item) ~= 'table' then complete = false; break end
                item.vector = nil
                -- Retain context only at the start of the selected contiguous tail.
                local context = item.recent_context
                item.recent_context = nil
                local encoded = cjson.encode(item)
                if used + #encoded > tonumber(ARGV[3]) then complete = false; break end
                used = used + #encoded
                taken = taken + 1
                table.insert(result, 1, item)
                result[1].recent_context = context
                if #result > 1 then result[2].recent_context = nil end
            end
            if #result > 0 and result[1].recent_context then
                local refs_size = #cjson.encode(result[1].recent_context)
                if used + refs_size <= tonumber(ARGV[3]) then used = used + refs_size
                else result[1].recent_context = nil; complete = false end
            end
            return result
        end
        for _, ref in ipairs(refs) do
            if string.sub(ref,1,#prefix) == prefix then
                local sid = string.sub(ref,#prefix+1)
                local queue = 'Tomte:consolidation:queue:' .. ref
                local processing = queue .. ':processing'
                local nq, np = redis.call('llen',queue), redis.call('llen',processing)
                total = total + nq + np
                if taken < tonumber(ARGV[2]) and used < tonumber(ARGV[3]) then
                    local epoch = ARGV[4] .. sid
                    redis.call('set',epoch,ARGV[5],'NX','EX',ARGV[6])
                    local batch = {session_id=sid,generation=redis.call('get',epoch),
                                   queued=take(queue,nq),identity={},processing={}}
                    if np <= tonumber(ARGV[2]) then
                        for i = 0, np-1 do
                            local ok,item = pcall(cjson.decode,redis.call('lindex',processing,i))
                            if not ok or type(item) ~= 'table' then
                                batch.identity = {}; complete = false; break
                            end
                            batch.identity[#batch.identity+1] = {item_id=item.item_id,
                                session_id=item.session_id,tenant_id=item.tenant_id,user_id=item.user_id}
                        end
                        if #batch.identity == np then batch.processing = take(processing,np) end
                    else complete = false end
                    sessions[#sessions+1] = batch
                elseif nq + np > 0 then complete = false end
            end
        end
        local anchors = {}
        local current = redis.call('hgetall',KEYS[1])
        for i = 1,#current,2 do
            local entry = cjson.decode(current[i+1])
            if entry.ready and entry.closed then anchors[#anchors+1] = entry end
        end
        return cjson.encode({sessions=sessions,anchors=anchors,complete=complete,
                             observed_messages=total,snapshot_at=now})
        """
        return json.loads(self._eval(script, [catalog,vectors,get_section_key_active_sessions()],
            f'{tenant}:{user}:', cfg.max_messages, cfg.max_input_bytes,
            f'Tomte:pending-search:{tenant}:{user}:epoch:', uuid4().hex, cfg.cache_ttl_seconds))

    def claim(self, tenant, user, session, generation, key, descriptor, owner):
        cfg = settings.pending_search
        script = _CACHE + """
        local raw = redis.call('hget',KEYS[1],ARGV[1])
        local entry = raw and cjson.decode(raw) or nil
        if entry and entry.ready then
            local vector = redis.call('hget',KEYS[2],ARGV[1])
            if vector then
                entry.used_at = now
                redis.call('hset',KEYS[1],ARGV[1],cjson.encode(entry))
                return {'ready',vector}
            end
        end
        if redis.call('get',KEYS[3]) ~= ARGV[2] then return {'retired'} end
        if entry and not entry.ready and entry.generation == ARGV[2] and entry.lease_until > now then
            return {'busy'}
        end
        if not entry and count >= tonumber(ARGV[7]) then
            if not oldest_key then return {'full'} end
            redis.call('hdel',KEYS[1],oldest_key)
            redis.call('hdel',KEYS[2],oldest_key)
        end
        entry = cjson.decode(ARGV[4])
        entry.generation, entry.owner = ARGV[2], ARGV[3]
        entry.lease_until, entry.expires_at = now+tonumber(ARGV[5]), now+tonumber(ARGV[6])
        entry.ready, entry.used_at, entry.nbytes = false, now, 0
        redis.call('hdel',KEYS[2],ARGV[1])
        redis.call('hset',KEYS[1],ARGV[1],cjson.encode(entry))
        redis.call('expire',KEYS[1],ARGV[6]); redis.call('expire',KEYS[2],ARGV[6])
        return {'owned'}
        """
        result = self._eval(script, keys(tenant,user,session), key,generation,owner,
                            json.dumps(descriptor),cfg.lease_seconds,cfg.cache_ttl_seconds,cfg.cache_windows)
        return {'state':result[0], **({'vector':json.loads(result[1])} if len(result)>1 else {})}

    def publish(self, tenant, user, session, generation, key, owner, vector):
        cfg = settings.pending_search
        script = _CACHE + """
        local raw = redis.call('hget',KEYS[1],ARGV[1])
        if not raw then return 0 end
        local entry = cjson.decode(raw)
        if entry.ready or entry.owner ~= ARGV[2] or entry.generation ~= ARGV[3]
            or redis.call('get',KEYS[3]) ~= ARGV[3] or entry.lease_until <= now then return 0 end
        if bytes + #ARGV[4] > tonumber(ARGV[5]) then
            redis.call('hdel',KEYS[1],ARGV[1]); return 0
        end
        entry.ready, entry.owner, entry.nbytes = true, nil, #ARGV[4]
        entry.expires_at, entry.used_at = now+tonumber(ARGV[6]), now
        redis.call('hset',KEYS[2],ARGV[1],ARGV[4])
        redis.call('hset',KEYS[1],ARGV[1],cjson.encode(entry))
        redis.call('expire',KEYS[1],ARGV[6]); redis.call('expire',KEYS[2],ARGV[6])
        return 1
        """
        return bool(self._eval(script,keys(tenant,user,session),key,owner,generation,
                               json.dumps(vector,separators=(',',':')),cfg.cache_bytes,cfg.cache_ttl_seconds))

    def release(self, tenant, user, session, key, owner):
        script = """
        if redis.call('type',KEYS[1]).ok ~= 'hash' then return 0 end
        local raw = redis.call('hget',KEYS[1],ARGV[1])
        if not raw then return 0 end
        local ok, entry = pcall(cjson.decode,raw)
        if ok and type(entry) == 'table' and not entry.ready and entry.owner == ARGV[2] then
            return redis.call('hdel',KEYS[1],ARGV[1])
        end
        return 0
        """
        return self._eval(script,keys(tenant,user,session),key,owner)
