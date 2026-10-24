"""Optional host-side CLI event timing. Never records reasoning content."""
import json
from pathlib import Path
import threading
import time


class EventTimeline:
    def __init__(self, path, started):
        self.path, self.started = Path(path), started
        self.events = []
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def start(self):
        self.thread.start()

    def _watch(self):
        try:
            with self.path.open() as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if line and line.endswith('\n'):
                        observed = time.monotonic() - self.started
                        try: event = json.loads(line)
                        except ValueError: continue
                        item = event.get('item') or {}
                        self.events.append(dict(observed_s=observed, type=event.get('type'),
                                                item_type=item.get('type'),
                                                usage=event.get('usage') if event.get('type') == 'turn.completed' else None))
                        continue
                    stream.seek(offset)
                    if self.stop.is_set(): break
                    self.stop.wait(.01)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive(): self.error = 'Observer did not finish'
        return self.events


def summarize(events, process_wall_s):
    def first(kind):
        return next((e['observed_s'] for e in events if e['type'] == kind), None)
    started, completed = first('turn.started'), first('turn.completed')
    return dict(first_event_s=events[0]['observed_s'] if events else None,
                thread_started_s=first('thread.started'), turn_started_s=started,
                turn_completed_s=completed,
                turn_window_s=completed-started if started is not None and completed is not None else None,
                post_turn_exit_s=max(0.,process_wall_s-completed) if completed is not None else None,
                definition='Host observes complete CLI JSON lines at 10 ms polling; includes transport/queue/model latency. No server-side inference-vs-network breakdown.')
