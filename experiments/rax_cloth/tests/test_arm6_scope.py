from pathlib import Path
from unittest.mock import Mock
import pytest
from rax_cloth.xarm_driver import configure_arm6,validate_request


def test_no_arm7_or_camb_tool_access():
    with pytest.raises(ValueError):validate_request('observe',{'camera':'camB'})
    with pytest.raises(ValueError):validate_request('move_joints',{
        'arm':'arm7','joints_deg':[0]*7,'observation_id':'fixture'})


def test_production_connect_only_opens_arm6(monkeypatch):
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.dual_arm import cli
    runtime=NativeRuntime(Path(__file__).resolve().parents[3]/'config/dual_arm.local.json')
    try:
        configure_arm6(runtime)
        factory=Mock();monkeypatch.setattr(cli,'XArmConnection',factory)
        connections=cli.connect(runtime.config,runtime.models,runtime.cancel,True)
        factory.assert_called_once()
        assert factory.call_args.args[0].axis==6
        assert connections['right'].static_obstacle
        assert connections['right'].snapshot()['live_feedback'] is False
        assert runtime.active_arms==('left',)
        assert getattr(runtime.config,'path_validation_enabled',True)
        assert not getattr(runtime.config,'collision_envelopes_disabled_arms',())
        with pytest.raises(Exception,match='cannot receive'):connections['right'].prepare_position()
    finally:runtime.close()
