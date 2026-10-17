"""Conservative classification of Codex stream reconnection diagnostics."""
import re

_RECONNECT = re.compile(r'Reconnecting\.\.\. \d+/\d+ \(stream disconnected before completion: .+\)', re.DOTALL)


def completion_status(events):
    """Only a later, final completed turn can recover a known reconnect notice.

    Exit code, nonempty final answer and resumed session identity must still be
    checked by the caller. Unknown errors and explicit turn.failed stay fatal.
    """
    boundaries = [(i, e['type']) for i, e in enumerate(events)
                  if e.get('type') in ('turn.started', 'turn.completed', 'turn.failed')]
    completed = bool(boundaries and boundaries[-1][1] == 'turn.completed')
    final_index = boundaries[-1][0] if completed else -1
    errors = []; recovered = []
    for i, event in enumerate(events):
        if event.get('type') not in ('error', 'turn.failed'): continue
        if (event.get('type') == 'error' and i < final_index
                and _RECONNECT.fullmatch(str(event.get('message', '')))):
            recovered.append(event)
        else:
            errors.append(event)
    return dict(final_turn_completed=completed, errors=errors, recovered_errors=recovered)
