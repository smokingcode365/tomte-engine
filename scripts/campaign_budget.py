"""Single-writer persistent budget for an explicitly approved evaluation campaign."""
from decimal import Decimal
import json
import math
from pathlib import Path
import time

from eval_api_budget import ApiBudget, BudgetStop


class CampaignBudget(ApiBudget):
    """A file lock and write-before-dispatch ledger prevent implicit budget resets.

    Approval is an operator record, not authentication. The person/agent running
    this tool must obtain real authorization before setting approved=True.
    """

    def __init__(self, directory, approval, pins):
        if approval.get('approved') is not True:
            raise ValueError('Campaign budget is not approved')
        if approval.get('pins') != pins:
            raise ValueError('Approval fingerprints differ from the current campaign')
        super().__init__(estimate=approval['estimate_usd'], ceiling=approval['ceiling_usd'],
                         approved_ceiling=approval['ceiling_usd'], request_limit=approval['request_limit'],
                         seconds=approval['active_seconds_limit'])
        self.directory = Path(directory)
        self.path = self.directory / 'budget.json'
        self.lock_path = self.directory / 'campaign.lock'
        self.pins = pins
        self.runs = {}
        self.phases = []
        self._handle = self.lock_path.open('x', encoding='utf-8')
        self._handle.write('Single writer. If interrupted, inspect ledger before removing this owned lock.\n')
        self._handle.flush()
        try:
            if self.path.exists():
                state = json.loads(self.path.read_text(encoding='utf-8'))
                if (state['pins'] != pins or Decimal(state['ceiling_usd']) != self.ceiling
                        or Decimal(state['estimate_usd']) != self.estimate
                        or state['request_limit'] != self.request_limit or state['active_seconds_limit'] != self.seconds):
                    raise ValueError('Existing campaign ledger has different bounds or fingerprints')
                if state['stopped'] or any(r['state'] != 'reconciled' for r in state['records']):
                    raise BudgetStop('Previous run stopped or has unresolved billing; review required')
                self.records = state['records']
                self.charged = Decimal(state['conservative_charged_usd'])
                self.usage_cost = Decimal(state['usage_cost_usd_without_discounts'])
                if (self.charged != sum(Decimal(r['conservative_cost_usd']) for r in self.records)
                        or self.usage_cost != sum(Decimal(r['usage_cost_usd']) for r in self.records)
                        or not math.isfinite(state['active_seconds']) or state['active_seconds'] < 0):
                    raise ValueError('Campaign ledger totals are inconsistent')
                self.started -= state['active_seconds']
                self.runs, self.phases = state['runs'], state['phases']
            self.checkpoint()
        except BaseException:
            self._release()
            raise

    def checkpoint(self):
        state = {**self.summary(), 'pins': self.pins, 'request_limit': self.request_limit,
                 'active_seconds_limit': self.seconds, 'active_seconds': time.monotonic() - self.started,
                 'runs': self.runs, 'phases': self.phases}
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.path)

    def reserve(self, url, payload):
        try:
            result = super().reserve(url, payload)
            self.checkpoint()  # Must succeed before bounded_http can dispatch.
            return result
        except BaseException:
            self.stopped = self.stopped or 'reservation_or_ledger_failure'
            self.checkpoint()
            raise

    def reconcile(self, ticket, response):
        try:
            return super().reconcile(ticket, response)
        finally:
            self.checkpoint()

    def begin_run(self, label):
        if label in self.runs:
            raise ValueError('Run was already attempted; no silent reruns')
        self.runs[label] = 'started'
        self.checkpoint()

    def end_run(self, label, success):
        assert self.runs[label] == 'started'
        self.runs[label] = 'completed' if success else 'failed'
        if not success:
            self.stopped = self.stopped or 'case_failure_review_required'
        self.checkpoint()

    def _release(self):
        self._handle.close()
        self.lock_path.unlink()  # Only the exact lock this instance created.

    def close(self):
        try:
            self.checkpoint()
        finally:
            self._release()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        if kind is not None and any(state == 'started' for state in self.runs.values()):
            self.stopped = self.stopped or 'campaign_interrupted'
        self.close()
