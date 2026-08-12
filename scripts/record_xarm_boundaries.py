import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from xarm.wrapper import XArmAPI


MEASUREMENTS = [
    ("x_min", 0, "X 最小安全位置"),
    ("y_min", 1, "Y 最小安全位置"),
    ("y_max", 1, "Y 最大安全位置"),
    ("z_min", 2, "Z 最低安全位置"),
    ("z_max", 2, "Z 最高安全位置"),
]


def read_robot(arm):
    code_p, pose = arm.get_position()
    if code_p != 0:
        raise RuntimeError(f"get_position() failed, code={code_p}")

    code_j, joints = arm.get_servo_angle()
    if code_j != 0:
        raise RuntimeError(f"get_servo_angle() failed, code={code_j}")

    return [float(v) for v in pose], [float(v) for v in joints]


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.200")
    parser.add_argument("--output", default="xarm_boundaries.json")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    arm = XArmAPI(args.ip)

    data = {
        "robot_ip": args.ip,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "boundary_mm": {},
        "samples": {},
    }

    try:
        if not arm.connected:
            raise RuntimeError(f"无法连接 xArm: {args.ip}")

        print(f"\n已连接 xArm: {args.ip}")
        print("state:", arm.get_state())
        print("error/warn:", arm.get_err_warn_code())

        print("\n即将进入 Free-Drive / Manual Mode。")
        print("请确认：")
        print("  1) 周围没有人或障碍物；")
        print("  2) 急停按钮在手边；")
        print("  3) 已安装夹爪时，负载/重力补偿设置合理；")
        print("  4) 用手扶住机械臂后再进入拖动模式。")
        input("\n准备好后按 Enter... ")

        # Unlock and switch to manual/free-drive mode.
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)

        ret = arm.set_mode(2)
        if ret != 0:
            raise RuntimeError(f"set_mode(2) failed, code={ret}")

        ret = arm.set_state(0)
        if ret != 0:
            raise RuntimeError(f"set_state(0) failed, code={ret}")

        time.sleep(0.5)

        print("\n已进入 Free-Drive。")
        print("坐标均为 xArm Base 坐标系，单位 mm。")
        print("每次只会取目标轴的值，另外两个轴的位置不会影响该边界记录。\n")

        for key, axis, description in MEASUREMENTS:
            print("=" * 66)
            print(f"测量 {key}: {description}")
            print("把末端拖到该安全边界。")
            cmd = input("到位后按 Enter 保存；输入 q 结束：").strip().lower()

            if cmd == "q":
                print("提前结束。")
                break

            # Let the arm settle after manual dragging.
            time.sleep(0.2)

            pose, joints = read_robot(arm)
            value = pose[axis]

            data["boundary_mm"][key] = round(value, 3)
            data["samples"][key] = {
                "tcp_pose_mm_deg": [round(v, 4) for v in pose],
                "joint_angles_deg": [round(v, 4) for v in joints],
                "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            }

            axis_name = "XYZ"[axis]
            print(
                f"保存成功：{key} = {value:.2f} mm\n"
                f"当前 TCP: X={pose[0]:.2f}, Y={pose[1]:.2f}, Z={pose[2]:.2f} mm"
            )

            # Save after every measurement in case the session is interrupted.
            save_json(output, data)

        print("\n" + "=" * 66)
        print("测量结果（原始值，尚未添加 safety margin）：")
        for key, _, _ in MEASUREMENTS:
            if key in data["boundary_mm"]:
                print(f"  {key:5s} = {data['boundary_mm'][key]:8.2f} mm")

        # Sanity checks
        b = data["boundary_mm"]
        if "y_min" in b and "y_max" in b and b["y_min"] >= b["y_max"]:
            print("\n⚠ 警告：y_min >= y_max，可能把两侧记录反了。")
        if "z_min" in b and "z_max" in b and b["z_min"] >= b["z_max"]:
            print("\n⚠ 警告：z_min >= z_max，可能把上下记录反了。")

        save_json(output, data)
        print(f"\n已保存到：{output}")
        print("这里只测量，没有把任何边界写入机器人控制器。")

        input("\n双手离开机械臂后按 Enter，切回 Mode 0... ")

    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，正在退出。", file=sys.stderr)

    finally:
        try:
            save_json(output, data)
        except Exception as e:
            print(f"保存文件失败: {e}", file=sys.stderr)

        try:
            if arm.connected:
                arm.set_mode(0)
                arm.set_state(0)
                time.sleep(0.2)
                arm.disconnect()
                print("已切回 Mode 0，并断开 SDK 连接。")
        except Exception as e:
            print(f"退出时切换模式失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

