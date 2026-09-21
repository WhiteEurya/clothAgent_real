"""Compact live progress for Claude CLI events; raw events are saved separately."""
from __future__ import annotations

from collections import Counter
import time


def urgent_event(event):
    subtype = str(event.get('subtype', '')).lower()
    return bool(event.get('is_error') or event.get('error') or event.get('errors') or
                event.get('type') == 'error' or
                any(word in subtype for word in ('error', 'failed', 'warning', 'retry')))


class ClaudeStreamProgress:
    """Batch status events every 5s and visible text excerpts every 1s.

    Event counts are not model turns. No text is inferred from thinking-token
    notifications, and no streamed proposal is treated as a final result.
    """

    def __init__(self, emit, *, clock=time.monotonic):
        self.emit, self.clock = emit, clock
        self.last_summary = self.last_text = self.clock()
        self.last_received = None
        self.counts = Counter()
        self.total = self.summarized = 0
        self.latest = None
        self.tools = {}
        self.tool_results = set()
        self.message_id = None
        self.partial_text = {}
        self.complete_text = set()
        self.pending_text = ''
        self.pending_characters = 0
        self.completed = False

    def _text(self, text):
        if isinstance(text, str) and text:
            self.pending_characters += len(text)
            self.pending_text = (self.pending_text + text)[-160:]

    def _partial(self, index, text):
        if isinstance(text, str) and text:
            key = (self.message_id, index)
            self.partial_text[key] = self.partial_text.get(key, '') + text
            self._text(text)

    def consume(self, event):
        kind, subtype = event.get('type', 'unknown'), event.get('subtype')
        self.latest = str(kind) + ('.' + str(subtype) if subtype else '')
        self.counts[self.latest] += 1
        self.total += 1
        self.last_received = self.clock()
        if kind == 'system' and subtype == 'responses_diagnostic':
            # `phase` is the first positional parameter of the fold pipeline's
            # progress callback. Keep the HTTP phase in a separate namespace.
            self.emit('responses_api', event.get('event', 'unknown'), event.get('elapsed_s'),
                      api_phase=event.get('phase'),
                      **{key: event.get(key) for key in ('category', 'http_status',
                         'client_request_id', 'headers_received_s', 'first_event_s',
                         'event_count', 'last_event', 'stream_requested', 'stream_fallback')})
        if kind == 'stream_event':
            part = event.get('event') or {}
            if part.get('type') == 'message_start':
                self.message_id = (part.get('message') or {}).get('id')
            elif part.get('type') == 'content_block_start':
                block = part.get('content_block') or {}
                if block.get('type') == 'text':
                    self._partial(part.get('index'), block.get('text'))
            elif part.get('type') == 'content_block_delta':
                delta = part.get('delta') or {}
                if delta.get('type') == 'text_delta':
                    self._partial(part.get('index'), delta.get('text'))
            # thinking_delta/signature_delta never become invented visible text.
            if urgent_event(part):
                self._notice(part)
        elif kind in {'assistant', 'user'}:
            message = event.get('message') or {}
            content = message.get('content', []) if isinstance(message, dict) else []
            if isinstance(content, str):
                content = [{'type': 'text', 'text': content}]
            for block in content:
                if not isinstance(block, dict):
                    continue
                if kind == 'assistant' and block.get('type') == 'text':
                    text = block.get('text', '')
                    identity = message.get('id', self.message_id)
                    key = (identity, text)
                    if identity is not None and key in self.complete_text:
                        continue
                    if identity is not None:
                        self.complete_text.add(key)
                    streamed = [value for (mid, _), value in self.partial_text.items()
                                if mid == identity and text.startswith(value)]
                    prefix = max(streamed, key=len, default='')
                    self._text(text[len(prefix):])
                elif block.get('type') == 'tool_use':
                    identity = block.get('id')
                    if identity and identity not in self.tools:
                        self.tools[identity] = block.get('name')
                        self._flush_text()
                        self.emit('claude_tool', 'started', tool=block.get('name'), tool_use_id=identity)
                elif block.get('type') == 'tool_result':
                    identity = block.get('tool_use_id')
                    if identity and identity not in self.tool_results:
                        self.tool_results.add(identity)
                        self.emit('claude_tool', 'failed' if block.get('is_error') else 'completed',
                                  tool=self.tools.get(identity), tool_use_id=identity)
        if kind == 'result':
            self.tick(force=True)
            self.completed = True
            self.emit('claude_result', 'failed' if urgent_event(event) else 'completed',
                      num_turns=event.get('num_turns'), stop_reason=event.get('stop_reason'),
                      subtype=subtype)
        elif urgent_event(event):
            self._notice(event)
        self.tick()

    def _notice(self, event):
        self._flush_text()
        subtype = str(event.get('subtype', '')).lower()
        failed = bool(event.get('is_error') or event.get('error') or event.get('errors') or
                      event.get('type') == 'error' or 'error' in subtype or 'failed' in subtype)
        self.emit('claude_notice', 'failed' if failed else 'received', message_type=event.get('type'),
                  subtype=event.get('subtype'),
                  detail=str(event.get('error') or event.get('errors') or event.get('message') or '')[:160])

    def _flush_text(self):
        if self.pending_characters:
            self.emit('claude_text', 'received', text_excerpt=self.pending_text,
                      characters=self.pending_characters,
                      excerpt_only=self.pending_characters > len(self.pending_text))
            self.pending_text = ''
            self.pending_characters = 0
            self.last_text = self.clock()

    def tick(self, *, force=False):
        now = self.clock()
        if force or now - self.last_text >= 1:
            self._flush_text()
        if self.completed or not self.total:
            return
        if force and self.total == self.summarized:
            return
        if force or now - self.last_summary >= 5:
            self.emit('claude_stream', 'progress', raw_events=self.total,
                      new_events=self.total - self.summarized,
                      event_counts=dict(self.counts), tool_calls=len(self.tools),
                      last_event=self.latest, idle_s=round(now - self.last_received, 1))
            self.last_summary, self.summarized = now, self.total
