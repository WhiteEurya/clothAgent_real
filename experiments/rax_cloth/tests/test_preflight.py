from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from rax_cloth.preflight import inspect

CONFIG = Path(__file__).resolve().parents[3]/'config/dual_arm.local.json'


def test_configuration_never_connects(tmp_path):
    factory = Mock(side_effect=AssertionError('must not connect'))
    report = inspect(CONFIG, tmp_path/'config', port_factory=factory)
    assert report['status'] == 'configuration_passed'
    factory.assert_not_called()


def test_live_diagnostic_only_reads_and_closes(tmp_path):
    port = Mock()
    port.state.return_value = {'status': 'test fixture'}
    port.capture.return_value = SimpleNamespace(meta={'observation_id': 'fixture', 'calibration_id': 'fixture'})
    report = inspect(CONFIG, tmp_path/'reads', True, Mock(return_value=port))
    assert report['status'] == 'live_reads_passed'
    assert [call.args[0] for call in port.capture.call_args_list] == ['camA']
    assert port.validate_observation.call_count == 1
    assert {call[0] for call in port.mock_calls} == {'state', 'capture', 'validate_observation', 'close'}


def test_read_failure_disconnects_without_motion(tmp_path):
    port = Mock()
    port.state.side_effect = RuntimeError('stale feedback fixture')
    report = inspect(CONFIG, tmp_path/'failure', True, Mock(return_value=port))
    assert report['status'] == 'failed'
    assert 'stale feedback' in report['error']
    assert {call[0] for call in port.mock_calls} == {'state', 'close'}
