from __future__ import annotations

from enum import Enum
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from cloth_agent.realsense_tuner import (
    CameraControls, PendingControls, TunerUI, build_parser, main, save_perception_values,
)


Option = Enum("Option", "enable_auto_exposure exposure gain enable_auto_white_balance white_balance temperature")


class Sensor:
    def __init__(self):
        self.values = dict(zip(Option, (1.0, 700.0, 32.0, 1.0, 3800.0, 37.0)))
        self.writes = []
        self.read_error = None
        self.write_error = None

    def get_supported_options(self):
        return list(Option)

    def get_option_range(self, option):
        lo, hi, step = {
            Option.enable_auto_exposure: (0, 1, 1),
            Option.exposure: (1, 10000, 1),
            Option.gain: (0, 128, 1),
            Option.enable_auto_white_balance: (0, 1, 1),
            Option.white_balance: (2800, 6500, 10),
            Option.temperature: (-40, 120, 1),
        }[option]
        return SimpleNamespace(min=lo, max=hi, step=step)

    def get_option(self, option):
        if option == self.read_error:
            raise RuntimeError("disconnected")
        return self.values[option]

    def set_option(self, option, value):
        if option == self.write_error:
            raise RuntimeError("device rejected option")
        self.values[option] = value
        self.writes.append((option, value))

    def is_option_read_only(self, option):
        return option == Option.temperature

    def get_option_description(self, option):
        return option.name


@pytest.fixture
def controls():
    return CameraControls(Sensor(), serial="camera-A", sensor_name="RGB Camera")


@pytest.mark.parametrize("name,auto,value", [
    ("exposure", Option.enable_auto_exposure, 1000),
    ("gain", Option.enable_auto_exposure, 64),
    ("white_balance", Option.enable_auto_white_balance, 4200),
])
def test_manual_adjustment_disables_auto_first_and_reads_hardware(controls, name, auto, value):
    assert controls.set_value(name, value) == value
    assert controls.sensor.writes == [(auto, 0), (Option[name], value)]


def test_invalid_and_readonly_values_do_not_touch_hardware(controls):
    for name, value in (("exposure", 0), ("gain", float("nan")), ("white_balance", 9000),
                        ("temperature", 35), ("unknown", 1)):
        with pytest.raises(ValueError):
            controls.set_value(name, value)
    assert controls.sensor.writes == []
    assert controls.set_value("white_balance", 3833) == 3830


def test_preset_roundtrip_restores_auto_and_omits_readonly(controls):
    preset = controls.preset()
    assert "temperature" not in preset["options"]
    controls.set_value("exposure", 1000)
    controls.set_value("white_balance", 4200)
    controls.apply_preset(preset)
    assert controls.preset()["options"] == preset["options"]
    assert controls.sensor.values[Option.enable_auto_exposure] == 1
    assert controls.sensor.values[Option.enable_auto_white_balance] == 1


@pytest.mark.parametrize("field,value", [("serial", "camera-B"), ("sensor", "Stereo Module"), ("version", 2)])
def test_preset_identity_checked_before_write(controls, field, value):
    preset = controls.preset()
    preset[field] = value
    with pytest.raises(ValueError):
        controls.apply_preset(preset)
    assert controls.sensor.writes == []


def test_entire_preset_validated_before_first_write(controls):
    preset = controls.preset()
    preset["options"]["white_balance"] = -1
    with pytest.raises(ValueError):
        controls.apply_preset(preset)
    assert controls.sensor.writes == []


def test_failed_read_never_saves_cached_values(controls):
    controls.sensor.read_error = Option.exposure
    with pytest.raises(RuntimeError, match="cannot save stale readings"):
        controls.preset()


def test_drag_events_coalesce_and_keep_latest_auto_manual_order():
    pending = PendingControls()
    pending.put("exposure", 800)
    pending.put("gain", 10)
    pending.put("exposure", 900)
    pending.put("enable_auto_exposure", 1)
    assert list(pending.take().items()) == [("gain", 10), ("exposure", 900), ("enable_auto_exposure", 1)]
    assert pending.take() == {}


def test_config_save_is_explicit_scoped_and_backed_up(controls, tmp_path):
    path = tmp_path / "perception.json"
    config = {"other": "keep", "cameras": [
        {"label": "A", "serial": "camera-A", "extrinsics_file": "original.yaml"},
        {"label": "B", "serial": "camera-B", "color_exposure": 400},
    ]}
    original = json.dumps(config)
    path.write_text(original)
    with pytest.raises(ValueError, match="关闭"):
        save_perception_values(path, controls)
    assert path.read_text() == original
    controls.set_value("exposure", 1200)
    controls.set_value("white_balance", 4300)
    backup = save_perception_values(path, controls)
    assert backup.read_text() == original
    updated = json.loads(path.read_text())
    assert updated["other"] == config["other"]
    assert updated["cameras"][1] == config["cameras"][1]
    assert updated["cameras"][0] == {
        **config["cameras"][0], "color_exposure": 1200, "color_white_balance": 4300,
    }


def test_ui_rejected_write_reports_error_and_keeps_preview_alive(controls, tmp_path):
    server = MagicMock()
    server.gui.add_markdown.side_effect = lambda *args, **kwargs: MagicMock()
    ui = TunerUI(server, controls, width=640, height=480, config_path=None, output_dir=tmp_path)
    controls.sensor.write_error = Option.exposure
    ui.pending.put("exposure", 900)
    assert ui.process_requests()
    assert "device rejected option" in ui.status.content
    ui.actions.put("stop")
    assert not ui.process_requests()


def test_ui_slider_callback_does_not_write_on_viser_programmatic_readback(controls, tmp_path):
    server = MagicMock()
    ui = TunerUI(server, controls, width=640, height=480, config_path=None, output_dir=tmp_path)
    # Exposure is the first slider. Programmatic update has client_id=None.
    callback = server.gui.add_slider.return_value.on_update.call_args_list[0].args[0]
    callback(SimpleNamespace(client_id=None, target=SimpleNamespace(value=900)))
    assert ui.pending.take() == {}
    callback(SimpleNamespace(client_id=42, target=SimpleNamespace(value=900)))
    assert controls.sensor.writes == []
    assert ui.pending.take() == {"exposure": 900}


def test_snapshot_saves_original_pixels_and_never_overwrites(controls, tmp_path):
    import time
    from PIL import Image

    ui = TunerUI(MagicMock(), controls, width=640, height=480, config_path=None, output_dir=tmp_path)
    ui.last_frame = np.full((480, 640, 3), [21, 100, 200], dtype=np.uint8)
    ui.last_frame_time = time.monotonic()
    ui._save(True)
    ui._save(True)
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert len(list(tmp_path.glob("*.png"))) == 2
    for path in tmp_path.glob("*.json"):
        preset = json.loads(path.read_text())
        assert preset["serial"] == "camera-A"
        pixels = np.asarray(Image.open(tmp_path / preset["image"]))
        np.testing.assert_array_equal(pixels, ui.last_frame)
    ui.last_frame_time -= 3
    with pytest.raises(ValueError, match="新鲜画面"):
        ui._save(True)


def test_main_stops_stream_and_server_on_interrupt(monkeypatch, tmp_path):
    import cloth_agent.realsense_tuner as module

    sensor = Sensor()
    sensor.get_info = lambda _: "RGB Camera"
    rs = MagicMock()
    pipeline = rs.pipeline.return_value
    pipeline.start.return_value.get_device.return_value.first_color_sensor.return_value = sensor
    pipeline.try_wait_for_frames.side_effect = KeyboardInterrupt
    viser = MagicMock()
    ui = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "pyrealsense2", rs)
    monkeypatch.setitem(__import__("sys").modules, "viser", viser)
    monkeypatch.setattr(module, "TunerUI", ui)
    assert main(["--project-root", str(tmp_path), "--serial", "camera-A"]) == 0
    pipeline.stop.assert_called_once()
    viser.ViserServer.return_value.stop.assert_called_once()
    assert build_parser().parse_args([]).camera == "A"


def test_failed_preset_load_releases_camera(monkeypatch, tmp_path):
    sensor = Sensor()
    sensor.get_info = lambda _: "RGB Camera"
    rs = MagicMock()
    pipeline = rs.pipeline.return_value
    pipeline.start.return_value.get_device.return_value.first_color_sensor.return_value = sensor
    preset = tmp_path / "wrong.json"
    preset.write_text('{"version": 0}')
    monkeypatch.setitem(__import__("sys").modules, "pyrealsense2", rs)
    monkeypatch.setitem(__import__("sys").modules, "viser", MagicMock())
    assert main(["--project-root", str(tmp_path), "--serial", "camera-A", "--preset", str(preset)]) == 1
    pipeline.stop.assert_called_once()
    assert sensor.writes == []
