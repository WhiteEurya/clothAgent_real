"""Record only the two lateral TCP boundaries; Z is configured separately."""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloth_agent.config import WorkspaceBounds

DEFAULT_OUTPUT = "data/robot/xarm_boundaries_new.json"


def read_robot(arm):
    code_p, pose = arm.get_position()
    code_j, joints = arm.get_servo_angle()
    if code_p != 0 or code_j != 0:
        raise RuntimeError(f"读取位姿失败: position={code_p}, joints={code_j}")
    return [float(v) for v in pose], [float(v) for v in joints]


def save_component(args, updates, samples, remove=()):
    """Merge with the latest output so independent captures preserve each other."""
    output = Path(args.output)
    source = output if output.exists() else getattr(args, "base_boundaries", None)
    data = json.loads(Path(source).read_text()) if source else {}
    if "boundary_mm" not in data:
        data = {"boundary_mm": data}
    bounds = data["boundary_mm"]
    for key in remove:
        bounds.pop(key, None)
        data.get("samples", {}).pop(key, None)
    bounds.update(updates)
    validated = WorkspaceBounds.from_mapping(bounds)
    data.setdefault("samples", {}).update(samples)
    data["robot_ip"] = args.ip
    data["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    if output.exists():
        backup = output.with_name(output.name + datetime.now().strftime(".%Y%m%dT%H%M%S%f.bak"))
        backup.write_bytes(output.read_bytes())
        print(f"旧文件已备份：{backup}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"已保存：{output}")
    if not validated.complete:
        print("边界尚未完整：左右两侧和 Z 下限均设置后，才可用于真实运行。")


def capture_points(args, prompts):
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(args.ip, is_radian=False)
    samples = {}
    try:
        if not arm.connected:
            raise RuntimeError(f"无法连接 xArm: {args.ip}")
        for name, label in prompts:
            if input(f"将 TCP 移到{label}并停稳，回车记录；q 取消：").strip().lower() == "q":
                print("已取消，未修改任何边界文件。")
                return None
            pose, joints = read_robot(arm)
            samples[name] = {"tcp_pose_mm_deg": pose, "joint_angles_deg": joints,
                             "timestamp": datetime.now().astimezone().isoformat(timespec="seconds")}
            print(f"已记录 TCP：{pose[:3]} mm")
    finally:
        arm.disconnect()
    return samples


def record_sides(args):
    print("左右选点：用 UFactory 手动操作机械臂，本程序只读取位置。")
    print("保持夹爪朝向，沿左右方向选两个 TCP 安全极限点；两点顺序不限。")
    print("只更新左右两侧，不设置或修改 Z 高度。")
    samples = capture_points(args, (("side_1", "第一侧安全极限"), ("side_2", "另一侧安全极限")))
    if samples is None:
        return
    points = [samples[k]["tcp_pose_mm_deg"][:2] for k in ("side_1", "side_2")]
    updates = {"lateral_points_mm": points}
    nx, ny, low, high = WorkspaceBounds.from_mapping(updates).lateral_geometry()
    print(f"左右允许宽度：{high-low:.2f} mm；限制方向：({nx:.4f}, {ny:.4f})")
    if input("输入 SAVE 保存两侧边界，其他输入取消：").strip() != "SAVE":
        print("已取消，未修改任何边界文件。")
        return
    save_component(args, updates, samples, remove=("x_min", "x_max", "y_min", "y_max"))


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ip", default="192.168.2.232")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--base-boundaries", default=None,
                        help="仅在输出文件不存在时，从指定文件保留其他边界；默认新建")
    return parser


def main():
    record_sides(build_parser(__doc__).parse_args())


if __name__ == "__main__":
    main()
