import pytest

from cloth_agent.robot_api import _connect_xarm, RobotExecutionError


@pytest.mark.parametrize('failures', [0, 1, 3])
def test_connection_retries_cleanup_and_bound(monkeypatch, failures):
    events = []
    sleeps = []
    monkeypatch.setattr('cloth_agent.robot_api.time.sleep', sleeps.append)

    class Arm:
        connected = False

        def __init__(self, ip, *, do_not_open):
            assert ip == 'test-arm' and do_not_open
            self.number = sum(e[0] == 'create' for e in events)
            events.append(('create', self.number))

        def connect(self):
            events.append(('connect', self.number))
            if self.number < failures:
                raise Exception('connect socket failed')
            self.connected = True

        def disconnect(self):
            events.append(('disconnect', self.number))

    if failures == 3:
        with pytest.raises(RobotExecutionError, match='test-arm.*3 attempt'):
            _connect_xarm(Arm, 'test-arm')
    else:
        assert _connect_xarm(Arm, 'test-arm').connected
    assert sum(e[0] == 'disconnect' for e in events) == failures
    assert len(sleeps) == min(failures, 2)
    for i, event in enumerate(events):
        if event[0] == 'create' and event[1] > 0:
            assert events[i - 1] == ('disconnect', event[1] - 1)


def test_non_transport_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr('cloth_agent.robot_api.time.sleep', lambda _: pytest.fail('unexpected retry'))
    events = []

    class Arm:
        def __init__(self, *args, **kwargs): pass
        def connect(self): raise ValueError('invalid configuration')
        def disconnect(self): events.append('closed')

    with pytest.raises(RobotExecutionError, match='1 attempt.*invalid configuration'):
        _connect_xarm(Arm, 'test-arm')
    assert events == ['closed']
