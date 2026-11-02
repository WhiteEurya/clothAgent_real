"""Durable reservation accounting for a future host-owned LLM relay.

No provider is called here. On crash, reservations remain charged (fail closed).
The caller must supply a conservative priced upper bound before billable I/O.
"""
import math
import time
import uuid


class Budget:
    def __init__(self, session):
        self.session = session

    def reserve(self, maximum_usd, *, phase, model):
        s = self.session.state
        if type(maximum_usd) not in (int, float) or not math.isfinite(maximum_usd) or maximum_usd <= 0:
            raise ValueError('positive finite conservative cost bound required')
        if s['llm_calls'] >= s['config']['max_llm_calls']:
            raise PermissionError('LLM call budget exhausted')
        if s['llm_cost_usd'] + maximum_usd > s['config']['budget_usd']:
            raise PermissionError('LLM cost budget exhausted')
        ident = uuid.uuid4().hex
        record = dict(maximum_usd=maximum_usd, phase=phase, model=model, started_unix_s=time.time())
        s.setdefault('reservations', {})[ident] = record
        s['llm_calls'] += 1
        s['llm_cost_usd'] += maximum_usd
        self.session.save()
        self.session.audit('llm_reserved', call_id=ident, **record)
        return ident

    def settle(self, ident, *, actual_usd, input_tokens, output_tokens, elapsed_seconds, response):
        s = self.session.state
        r = s['reservations'][ident]
        if type(actual_usd) not in (int, float) or not math.isfinite(actual_usd) or not 0 <= actual_usd <= r['maximum_usd']:
            raise ValueError('provider exceeded reserved bound; retain full reservation and stop')
        if any(type(x) is not int or x < 0 for x in (input_tokens, output_tokens)):
            raise ValueError('actual token usage required')
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError('measured elapsed time required')
        # Provider-visible answer/reasoning summary/tool calls only; no invented hidden reasoning.
        self.session.audit('llm_result', call_id=ident, model=r['model'], phase=r['phase'],
                           actual_usd=actual_usd, input_tokens=input_tokens, output_tokens=output_tokens,
                           elapsed_seconds=elapsed_seconds, response=response)
        del s['reservations'][ident]
        s['llm_cost_usd'] -= r['maximum_usd'] - actual_usd
        s['input_tokens'] += input_tokens
        s['output_tokens'] += output_tokens
        self.session.save()
