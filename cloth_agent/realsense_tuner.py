"""Live RGB controls for a RealSense camera; all SDK access stays on one thread."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from .run_storage import auxiliary_dir
import queue
import threading
import time
from typing import Any


AUTO_OPTIONS = {"enable_auto_exposure", "enable_auto_white_balance"}
MANUAL_AUTO = {
    "exposure": "enable_auto_exposure",
    "gain": "enable_auto_exposure",
    "white_balance": "enable_auto_white_balance",
}
COMMON_OPTIONS = (
    "enable_auto_exposure", "exposure", "gain",
    "enable_auto_white_balance", "white_balance", "brightness", "contrast",
    "saturation", "sharpness", "gamma", "backlight_compensation", "power_line_frequency",
)
LABELS = {
    "enable_auto_exposure": "自动曝光",
    "exposure": "曝光 Exposure",
    "gain": "增益 Gain",
    "enable_auto_white_balance": "自动白平衡",
    "white_balance": "白平衡 White balance",
    "brightness": "亮度 Brightness",
    "contrast": "对比度 Contrast",
    "saturation": "饱和度 Saturation",
    "sharpness": "锐度 Sharpness",
    "gamma": "Gamma",
    "backlight_compensation": "背光补偿",
    "power_line_frequency": "电源频率 / 防闪烁",
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_json(path: Path, data: Any) -> None:
    # Timestamped outputs must never overwrite an earlier measurement.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


@dataclass(frozen=True)
class OptionSpec:
    sdk_option: Any
    minimum: float
    maximum: float
    step: float
    readonly: bool
    description: str
    choices: dict[str, float]

    def normalize(self, value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"not a numeric option value: {value!r}") from exc
        if not math.isfinite(value) or not self.minimum <= value <= self.maximum:
            raise ValueError(f"{value} outside [{self.minimum}, {self.maximum}]")
        if self.step > 0:
            value = self.minimum + round((value - self.minimum) / self.step) * self.step
        return min(self.maximum, max(self.minimum, value))


class CameraControls:
    def __init__(self, sensor: Any, *, serial: str, sensor_name: str) -> None:
        self.sensor, self.serial, self.sensor_name = sensor, serial, sensor_name
        self.specs: dict[str, OptionSpec] = {}
        self.values: dict[str, float] = {}
        self.discovery_errors: list[str] = []
        for option in sensor.get_supported_options():
            name = option.name
            try:
                limits = sensor.get_option_range(option)
                value = float(sensor.get_option(option))
                if not all(math.isfinite(v) for v in (limits.min, limits.max, limits.step, value)):
                    raise ValueError("non-finite option range or value")
                choices = {}
                if limits.step == 1 and 1 < limits.max - limits.min <= 16:
                    for candidate in range(math.ceil(limits.min), math.floor(limits.max) + 1):
                        try:
                            label = sensor.get_option_value_description(option, candidate)
                        except RuntimeError:
                            label = None
                        if not label:
                            choices = {}
                            break
                        choices[f"{candidate}: {label}"] = float(candidate)
                self.specs[name] = OptionSpec(
                    option, limits.min, limits.max, limits.step,
                    sensor.is_option_read_only(option) or limits.min == limits.max,
                    sensor.get_option_description(option), choices,
                )
                self.values[name] = value
            except (RuntimeError, ValueError) as exc:
                self.discovery_errors.append(f"{name}: {exc}")

    def validate(self, name: str, value: float) -> float:
        if name not in self.specs:
            raise ValueError(f"camera does not support {name}")
        spec = self.specs[name]
        if spec.readonly:
            raise ValueError(f"{name} is read-only")
        return spec.normalize(value)

    def set_value(self, name: str, value: float) -> float:
        value = self.validate(name, value)
        automatic = MANUAL_AUTO.get(name)
        if automatic in self.specs:
            auto_spec = self.specs[automatic]
            if self.sensor.get_option(auto_spec.sdk_option) != 0:
                self.validate(automatic, 0)
                self.sensor.set_option(auto_spec.sdk_option, 0)
        self.sensor.set_option(self.specs[name].sdk_option, value)
        measured = float(self.sensor.get_option(self.specs[name].sdk_option))
        self.values[name] = measured
        return measured

    def refresh(self) -> list[str]:
        errors = []
        for name, spec in self.specs.items():
            try:
                self.values[name] = float(self.sensor.get_option(spec.sdk_option))
            except RuntimeError as exc:
                errors.append(f"{name}: {exc}")
        return errors

    def preset(self) -> dict[str, Any]:
        errors = self.refresh()
        if errors:
            raise RuntimeError("cannot save stale readings: " + "; ".join(errors))
        return {
            "version": 1,
            "kind": "realsense_rgb_controls",
            "saved_at_utc": _stamp(),
            "serial": self.serial,
            "sensor": self.sensor_name,
            "options": {name: self.values[name] for name, spec in self.specs.items() if not spec.readonly},
        }

    def apply_preset(self, payload: dict[str, Any]) -> None:
        if payload.get("version") != 1 or payload.get("kind") != "realsense_rgb_controls":
            raise ValueError("not a RealSense RGB controls preset")
        if payload.get("serial") != self.serial or payload.get("sensor") != self.sensor_name:
            raise ValueError("preset camera serial/sensor does not match the connected camera")
        options = payload.get("options")
        if not isinstance(options, dict) or not options:
            raise ValueError("preset has no options")
        # Validate the entire file before changing hardware. Manual values can
        # switch auto off; apply saved auto flags LAST so auto-on presets work.
        values = {name: self.validate(name, value) for name, value in options.items()}
        for name in sorted(values, key=lambda key: key in AUTO_OPTIONS):
            self.set_value(name, values[name])


class PendingControls:
    """Coalesce rapid drag events; callbacks never call the camera SDK."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.values: dict[str, float] = {}

    def put(self, name: str, value: float) -> None:
        with self.lock:
            # Move the latest event to the end to preserve auto/manual order.
            self.values.pop(name, None)
            self.values[name] = value

    def take(self) -> dict[str, float]:
        with self.lock:
            values, self.values = self.values, {}
            return values


def save_perception_values(path: Path, controls: CameraControls) -> Path:
    """Explicit UI action: persist ONLY the two supported perception fields."""
    preset = controls.preset()
    options = preset["options"]
    if any(options.get(name, 0) != 0 for name in AUTO_OPTIONS):
        raise ValueError("请先关闭自动曝光和自动白平衡，再写入折叠配置")
    if not {"exposure", "white_balance"} <= options.keys():
        raise ValueError("相机没有可保存的曝光 / 白平衡读数")
    original = path.read_text(encoding="utf-8")
    config = json.loads(original)
    matches = [c for c in config["cameras"] if c["serial"] == controls.serial]
    if len(matches) != 1:
        raise ValueError("配置中必须恰好有一台相机与当前序列号相同")
    matches[0]["color_exposure"] = options["exposure"]
    matches[0]["color_white_balance"] = options["white_balance"]
    backup = path.with_name(f"{path.name}.{_stamp()}.bak")
    with backup.open("x", encoding="utf-8") as handle:
        handle.write(original)
    temporary = path.with_name(f".{path.name}.{_stamp()}.tmp")
    _write_json(temporary, config)
    temporary.replace(path)
    return backup


class TunerUI:
    def __init__(self, server: Any, controls: CameraControls, *, width: int, height: int,
                 config_path: Path | None, output_dir: Path) -> None:
        import numpy as np

        self.server, self.controls = server, controls
        self.pending = PendingControls()
        self.actions: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.handles: dict[str, Any] = {}
        self.config_path, self.output_dir = config_path, output_dir
        self.startup_preset = controls.preset()
        self.last_frame = None
        self.last_frame_time = 0.0
        server.gui.configure_theme(control_layout="collapsible", control_width="large",
                                   show_share_button=False)
        server.scene.world_axes.visible = False
        self.preview = server.scene.add_image(
            "/RGB", np.zeros((height, width, 3), dtype=np.uint8),
            render_width=width / height, render_height=1.0,
            format="jpeg", jpeg_quality=90, cast_shadow=False, receive_shadow=False,
        )

        def reset_view(client: Any) -> None:
            client.camera.position = (0.0, 0.0, 2.0)
            client.camera.up_direction = (0.0, 1.0, 0.0)
            client.camera.look_at = (0.0, 0.0, 0.0)
            client.camera.fov = math.radians(45)

        server.on_client_connect(reset_view)
        server.gui.add_markdown(
            f"### RealSense RGB 调参\n相机 `{controls.serial}` · {width} × {height}\n\n"
            "拖动滑块即可生效；手动改曝光/增益会关闭自动曝光，改白平衡会关闭自动白平衡。"
            "数值采用 SDK 原生单位。只显示当前 RGB 传感器支持的参数。"
        )
        self.status = server.gui.add_markdown("正在等待相机画面…")
        self.frame_status = server.gui.add_markdown("")
        reset = server.gui.add_button("恢复画面视角")
        reset.on_click(lambda event: reset_view(event.client) if event.client is not None else None)
        with server.gui.add_folder("常用图像参数"):
            for name in COMMON_OPTIONS:
                if name in controls.specs and not controls.specs[name].readonly:
                    self._add_control(name)
        with server.gui.add_folder("其他 RGB 参数", expand_by_default=False):
            for name, spec in controls.specs.items():
                if name not in COMMON_OPTIONS and not spec.readonly:
                    self._add_control(name)
        with server.gui.add_folder("保存 / 恢复"):
            server.gui.add_markdown(
                f"完整参数与原分辨率 PNG 保存在 `{output_dir}`。预设用 `--preset 文件.json` 重新加载。"
            )
            for label, action in (("保存完整参数预设", "preset"), ("截图 + 参数", "snapshot"),
                                  ("恢复本次启动参数", "restore")):
                server.gui.add_button(label).on_click(lambda _, action=action: self.actions.put(action))
            if config_path is not None:
                server.gui.add_markdown(
                    f"下面按钮仅将**曝光、白平衡**写入 `{config_path.name}`，并备份原文件。"
                    "其他参数不会自动接入折叠流程；需先关闭两项自动控制。"
                )
                server.gui.add_button("写入折叠配置：曝光 / 白平衡").on_click(
                    lambda _: self.actions.put("config"))
        with server.gui.add_folder("相机实际读数 / 诊断", expand_by_default=False):
            self.readings = server.gui.add_markdown("")
            if controls.discovery_errors:
                server.gui.add_markdown("无法查询的参数：\n\n" + "\n\n".join(controls.discovery_errors))
        server.gui.add_button("停止相机并退出", color="red").on_click(lambda _: self.actions.put("stop"))
        self.sync_controls()

    def _add_control(self, name: str) -> None:
        spec = self.controls.specs[name]
        value = self.controls.values[name]
        label = LABELS.get(name, name)
        hint = f"{spec.description} [{spec.minimum:g}, {spec.maximum:g}], step={spec.step:g}"
        if spec.minimum == 0 and spec.maximum == 1 and spec.step == 1:
            handle = self.server.gui.add_checkbox(label, bool(value), hint=hint)
        elif spec.choices:
            selected = next((key for key, v in spec.choices.items() if v == value), None)
            handle = self.server.gui.add_dropdown(label, tuple(spec.choices), initial_value=selected, hint=hint)
        else:
            handle = self.server.gui.add_slider(
                label, min=spec.minimum, max=spec.maximum,
                step=spec.step or (spec.maximum - spec.minimum) / 1000,
                initial_value=min(spec.maximum, max(spec.minimum, value)), hint=hint,
            )

        @handle.on_update
        def changed(event: Any) -> None:
            # Viser also invokes callbacks for programmatic readback updates.
            if event.client_id is not None:
                value = event.target.value
                self.pending.put(name, spec.choices[value] if spec.choices else float(value))

        self.handles[name] = handle

    def message(self, text: str) -> None:
        self.status.content = text
        print(f"[camera-tuner] {text}", flush=True)

    def sync_controls(self) -> None:
        errors = self.controls.refresh()
        rows = ["| 参数 | 实际读数 |", "| --- | ---: |"]
        for name, spec in self.controls.specs.items():
            value = self.controls.values[name]
            rows.append(f"| {name}{' (只读)' if spec.readonly else ''} | {value:g} |")
            handle = self.handles.get(name)
            if handle is None:
                continue
            if spec.choices:
                selected = next((key for key, v in spec.choices.items() if v == value), None)
                if selected is not None:
                    handle.value = selected
            elif spec.minimum == 0 and spec.maximum == 1 and spec.step == 1:
                handle.value = bool(value)
            else:
                handle.value = min(spec.maximum, max(spec.minimum, value))
        self.readings.content = "\n".join(rows) + ("\n\n读取失败：" + "; ".join(errors) if errors else "")

    def process_requests(self) -> bool:
        pending = self.pending.take()
        for name, value in pending.items():
            try:
                actual = self.controls.set_value(name, value)
                self.message(f"{name}: 请求 {value:g} → 实际 {actual:g}")
            except (RuntimeError, ValueError) as exc:
                self.message(f"设置 {name} 失败：{exc}")
        if pending:
            self.sync_controls()
        while not self.actions.empty():
            action = self.actions.get_nowait()
            if action == "stop":
                return False
            try:
                if action == "restore":
                    self.controls.apply_preset(self.startup_preset)
                    self.message("已恢复本次启动参数")
                elif action == "config":
                    backup = save_perception_values(self.config_path, self.controls)
                    self.message(f"已写入曝光 / 白平衡。原配置备份：{backup}")
                else:
                    self._save(action == "snapshot")
            except (RuntimeError, ValueError, OSError) as exc:
                self.message(f"操作失败：{exc}")
            self.sync_controls()
        return True

    def _save(self, snapshot: bool) -> None:
        from PIL import Image

        if snapshot and (self.last_frame is None or time.monotonic() - self.last_frame_time > 2):
            raise ValueError("没有新鲜画面，请等待相机恢复后再截图")
        payload = self.controls.preset()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.controls.serial}_{_stamp()}"
        path = self.output_dir / f"{stem}.json"
        if snapshot:
            image_path = self.output_dir / f"{stem}.png"
            Image.fromarray(self.last_frame).save(image_path)
            payload["image"] = image_path.name
            payload["frame_age_s"] = time.monotonic() - self.last_frame_time
        _write_json(path, payload)
        self.message(f"已保存：{path}" + (" 和 PNG 原图" if snapshot else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RealSense RGB 实时预览与滑块调参（不连接机器人）")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--perception-config", type=Path, default=Path("config/perception.free_exploration.json"))
    parser.add_argument("--camera", default="A", help="配置中的相机标签，默认 A")
    parser.add_argument("--serial", help="直接指定相机序列号，可不依赖感知配置")
    parser.add_argument("--list", action="store_true", help="列出已连接相机并退出")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--preview-fps", type=float, default=10, help="浏览器画面刷新率，默认 10")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument("--preset", type=Path, help="加载之前保存的完整参数，必须匹配相机序列号")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        import pyrealsense2 as rs
    except ImportError:
        parser.exit(1, "缺少 pyrealsense2，请使用装有 RealSense SDK 的 cali 环境运行。\n")
    if args.list:
        try:
            devices = list(rs.context().query_devices())
        except RuntimeError as exc:
            parser.exit(1, f"无法枚举 RealSense 相机：{exc}；请检查 USB 连接及设备访问权限。\n")
        for device in devices:
            print(device.get_info(rs.camera_info.serial_number), device.get_info(rs.camera_info.name))
        if not devices:
            print("未发现 RealSense 相机，请检查 USB 连接和设备访问权限。")
        return 0

    root = args.project_root.expanduser().resolve()
    config_path = (root / args.perception_config).resolve()
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    selected = next((c for c in config.get("cameras", []) if (
        c["serial"] == args.serial if args.serial else c["label"].upper() == args.camera.upper()
    )), None)
    if selected is None and not args.serial:
        parser.error(f"在 {config_path} 中找不到相机 {args.camera}；可用 --list 和 --serial 选择设备")
    selected = selected or {}
    serial = args.serial or selected["serial"]
    width = args.width if args.width is not None else config.get("width", 1280)
    height = args.height if args.height is not None else config.get("height", 720)
    fps = args.fps if args.fps is not None else config.get("fps", 30)
    if min(width, height, fps) <= 0 or not math.isfinite(args.preview_fps) or args.preview_fps <= 0:
        parser.error("width / height / fps / preview-fps 必须为正数")
    if not 1 <= args.port <= 65535:
        parser.error("port 必须在 1 到 65535 之间")

    import numpy as np
    import viser
    from PIL import Image

    pipeline = None
    started = False
    server = None
    try:
        print(f"打开 RealSense {serial}: RGB {width}×{height} @ {fps} FPS", flush=True)
        pipeline, stream_config = rs.pipeline(), rs.config()
        stream_config.enable_device(serial)
        stream_config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = pipeline.start(stream_config)
        started = True
        sensor = profile.get_device().first_color_sensor()
        controls = CameraControls(sensor, serial=serial, sensor_name=sensor.get_info(rs.camera_info.name))
        if args.preset:
            controls.apply_preset(json.loads(args.preset.expanduser().read_text()))
        else:
            for key, name in (("color_exposure", "exposure"), ("color_white_balance", "white_balance")):
                if selected.get(key) is not None:
                    controls.set_value(name, selected[key])
        server = viser.ViserServer(host=args.host, port=args.port, label="RealSense RGB 调参")
        ui = TunerUI(server, controls, width=width, height=height,
                     config_path=config_path if selected else None,
                     output_dir=(root / args.output_dir).resolve() if args.output_dir else auxiliary_dir(root, "camera_tuning"))
        url_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        print(f"手动打开 http://{url_host}:{server.get_port()} 。Ctrl+C 退出；不会自动打开浏览器。", flush=True)
        last_preview = last_refresh = 0.0
        last_frame_at = time.monotonic()
        while ui.process_requests():
            received, frames = pipeline.try_wait_for_frames(timeout_ms=100)
            now = time.monotonic()
            frame = frames.get_color_frame() if received else None
            if frame:
                last_frame_at = now
                if now - last_preview >= 1 / args.preview_fps:
                    if ui.last_frame is None:
                        ui.message("已收到实时 RGB 画面，可以开始调参")
                    ui.last_frame = np.asanyarray(frame.get_data()).copy()
                    ui.last_frame_time = now
                    preview = Image.fromarray(ui.last_frame)
                    preview.thumbnail((960, 960))
                    ui.preview.image = np.asarray(preview)
                    last_preview = now
            if now - last_refresh >= 1:
                ui.sync_controls()
                age = now - last_frame_at
                ui.frame_status.content = (f"RGB 实时画面 · 最近一帧 {age:.1f} 秒前" if age < 2 else
                                           f"相机已 {age:.1f} 秒没有新画面，当前显示为旧帧。")
                last_refresh = now
            if now - last_frame_at > 15:
                raise RuntimeError("相机连续 15 秒没有新帧，请检查连接及所选分辨率 / 帧率")
    except KeyboardInterrupt:
        print("\n停止预览，释放相机。")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"相机调参失败：{exc}\n请检查相机连接、流格式，并先退出占用同一相机的折叠 / 采集程序。")
        return 1
    finally:
        try:
            if started:
                pipeline.stop()
        finally:
            if server is not None:
                server.stop()
    return 0
