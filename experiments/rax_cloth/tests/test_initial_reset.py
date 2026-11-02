import threading
import time
from unittest.mock import Mock
import pytest
from rax_cloth.initial_reset import run


def test_initial_reset_once_and_no_replay_after_completion(tmp_path):
    backend=Mock();backend._reset.return_value={'status':'EXECUTED','task_success':None}
    args=(tmp_path,backend,Mock(),threading.Event(),time.monotonic()+10,3,Mock())
    assert run(*args)['status']=='EXECUTED'
    assert run(*args)['status']=='EXECUTED'
    backend._reset.assert_called_once_with({'seed':3})


def test_failure_is_not_retried(tmp_path):
    backend=Mock();backend._reset.side_effect=RuntimeError('fixture proof failure')
    args=(tmp_path,backend,Mock(),threading.Event(),time.monotonic()+10,0,Mock())
    with pytest.raises(RuntimeError,match='initial reset failed'):run(*args)
    with pytest.raises(RuntimeError,match='incomplete'):run(*args)
    backend._reset.assert_called_once()


def test_no_reset_after_cancellation(tmp_path):
    stop=threading.Event();stop.set();backend=Mock()
    with pytest.raises(RuntimeError,match='before initial reset'):
        run(tmp_path,backend,Mock(),stop,time.monotonic()+10,0,Mock())
    backend._reset.assert_not_called()
