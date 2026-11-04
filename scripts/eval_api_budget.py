"""Bounded text-only evaluation transport. Never used by the engine at runtime."""
import asyncio
from contextlib import contextmanager
from decimal import Decimal
import json
from threading import Lock
import time

import httpx


class BudgetStop(RuntimeError):
    """No further external requests are authorized for this evaluation."""


class ApiBudget:
    # Standard short-context USD / million tokens, verified 2026-11-04.
    # Cache writes are conservatively charged at 1.25x ordinary input.
    RATES = {
        'gpt-6.1-sol': (Decimal('2'), Decimal('10')),
        'gpt-6-luna': (Decimal('.1'), Decimal('.5')),
        'text-embedding-3-small': (Decimal('.02'), Decimal('0')),
    }

    def __init__(self, *, ceiling='1.80', estimate='1.25', request_limit=200,
                 seconds=2700, output_tokens=8192):
        self.ceiling, self.estimate = Decimal(ceiling), Decimal(estimate)
        if not 0 < self.ceiling < 5 or not 0 < self.estimate <= self.ceiling:
            raise ValueError('Explicit estimate and ceiling below USD 5 required')
        self.request_limit, self.seconds, self.output_tokens = request_limit, seconds, output_tokens
        self.started = time.monotonic()
        self.charged = Decimal(0)
        self.usage_cost = Decimal(0)
        self.records = []
        self.stopped = None
        self.lock = Lock()

    def stop(self, reason):
        self.stopped = reason
        raise BudgetStop(reason)

    def reserve(self, url, payload):
        """Charge a conservative upper bound before dispatch; unknown fees fail closed."""
        with self.lock:
            if self.stopped:
                raise BudgetStop(self.stopped)
            if time.monotonic() - self.started >= self.seconds:
                self.stop('task_time_limit')
            if len(self.records) >= self.request_limit:
                self.stop('request_limit')
            url = httpx.URL(url)
            if url.scheme != 'https' or url.host != 'api.openai.com' or url.port not in (None, 443):
                self.stop('unpriced_endpoint')
            route = url.path
            body = dict(payload)
            model = body.get('model')
            if model not in self.RATES or route not in ('/v1/responses', '/v1/embeddings'):
                self.stop('unpriced_model_or_route')
            embedding = route.endswith('/embeddings')
            if embedding != (model == 'text-embedding-3-small'):
                self.stop('model_route_mismatch')
            if body.get('stream') or body.get('background') or body.get('previous_response_id'):
                self.stop('unbounded_response_state')
            if not embedding:
                if any(tool.get('type') != 'function' for tool in body.get('tools', [])):
                    self.stop('unpriced_builtin_tool')
                if body.get('prompt_cache_retention') or body.get('prompt'):
                    self.stop('unbounded_prompt_state')
                body['max_output_tokens'] = min(body.get('max_output_tokens', self.output_tokens), self.output_tokens)
                body['service_tier'] = 'default'
                body['store'] = False
            encoded = json.dumps(body, ensure_ascii=False).encode('utf-8')
            if len(encoded) > 96000:
                self.stop('request_byte_limit')
            # Serialized UTF-8 bytes bound text BPE token count; schema/message
            # framing receives an additional 8192-token allowance. No images/audio.
            if not embedding:
                inputs = body.get('input', [])
                if isinstance(inputs, list):
                    for item in inputs:
                        for content in item.get('content', []) if isinstance(item.get('content'), list) else []:
                            if content.get('type') not in ('input_text', 'output_text'):
                                self.stop('non_text_input')
            bound_input = len(encoded) + 8192
            bound_output = 0 if embedding else body['max_output_tokens']
            input_rate, output_rate = self.RATES[model]
            reservation = (Decimal(bound_input) * input_rate * Decimal('1.25')
                           + Decimal(bound_output) * output_rate) / Decimal(1000000)
            # Margin covers conservative rounding; no cached-input discounts assumed.
            reservation *= Decimal('1.10')
            effective_limit = min(self.ceiling, self.estimate * Decimal('1.5'))
            if self.charged + reservation > effective_limit:
                self.stop('spend_limit_before_dispatch')
            self.charged += reservation
            ticket = {'index': len(self.records), 'model': model, 'route': route,
                      'reservation_usd': str(reservation), 'input_token_bound': bound_input,
                      'output_token_bound': bound_output, 'state': 'reserved'}
            self.records.append(ticket)
            return body, ticket

    def reconcile(self, ticket, response):
        with self.lock:
            if ticket['state'] != 'reserved':
                raise ValueError('Reservation already reconciled')
            if response.status_code != 200:
                ticket.update(state='error_reservation_retained', http_status=response.status_code)
                self.stop('provider_http_error')
            data = response.json()
            usage = data.get('usage', {})
            input_tokens = usage.get('input_tokens', usage.get('prompt_tokens'))
            output_tokens = usage.get('output_tokens', 0 if ticket['route'].endswith('/embeddings') else None)
            if any(type(value) is not int or value < 0 for value in (input_tokens, output_tokens)):
                ticket['state'] = 'unknown_usage_reservation_retained'
                self.stop('missing_usage')
            if input_tokens > ticket['input_token_bound'] or output_tokens > ticket['output_token_bound']:
                self.stop('usage_exceeded_reservation_assumptions')
            if data.get('service_tier') not in (None, 'default'):
                self.stop('unexpected_service_tier')
            rate_in, rate_out = self.RATES[ticket['model']]
            # Usage-based standard-price estimate; billing discounts are not inferred.
            cost = (Decimal(input_tokens) * rate_in + Decimal(output_tokens) * rate_out) / Decimal(1000000)
            conservative = cost * Decimal('1.375')
            self.charged += conservative - Decimal(ticket['reservation_usd'])
            self.usage_cost += cost
            ticket.update(state='reconciled', usage=usage, usage_cost_usd=str(cost),
                          conservative_cost_usd=str(conservative), returned_model=data.get('model'))
            if self.usage_cost > self.estimate * Decimal('1.5'):
                self.stop('estimate_tolerance_exceeded')

    def summary(self):
        return {'estimate_usd': str(self.estimate), 'ceiling_usd': str(self.ceiling),
                'usage_cost_usd_without_discounts': str(self.usage_cost),
                'conservative_charged_usd': str(self.charged), 'requests': len(self.records),
                'warning_over_120_percent': self.usage_cost > self.estimate * Decimal('1.2'),
                'stopped': self.stopped, 'records': self.records}


@contextmanager
def bounded_http(budget, record):
    """Observe every HTTP attempt, including SDK retries; do not log headers."""
    original_async, original_sync = httpx.AsyncClient.send, httpx.Client.send

    async def send(client, request, *args, **kwargs):
        body, ticket = budget.reserve(request.url, json.loads(request.content))
        headers = dict(request.headers)
        headers.pop('content-length', None)
        revised = httpx.Request(request.method, request.url, headers=headers,
                               content=json.dumps(body, ensure_ascii=False).encode('utf-8'), extensions=request.extensions)
        started = time.monotonic()
        try:
            response = await asyncio.wait_for(original_async(client, revised, *args, **kwargs), timeout=120)
            await response.aread()
            record({'request_index': ticket['index'], 'request': body,
                    'response': response.json() if response.status_code == 200 else {'http_status': response.status_code},
                    'elapsed_seconds': time.monotonic() - started})
            budget.reconcile(ticket, response)
            return response
        except BaseException:
            budget.stopped = budget.stopped or 'transport_failure_unknown_billing'
            raise

    def reject_sync(*args, **kwargs):
        budget.stop('unexpected_sync_transport')

    httpx.AsyncClient.send, httpx.Client.send = send, reject_sync
    try:
        yield
    finally:
        httpx.AsyncClient.send, httpx.Client.send = original_async, original_sync
