"""Record only the minimum TCP height, preserving lateral bounds."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.record_xarm_boundaries import build_parser, capture_points, save_component
from cloth_agent.config import WorkspaceBounds


def record_z(args):
    print("Z 高度选点：用 UFactory 手动操作机械臂，本程序只读取位置。")
    print("只设置最低高度，不设高度上限，保留已经设置的左右边界。")
    samples = capture_points(args, (("z_min", "最低允许高度"),))
    if samples is None:
        return
    updates = {key: samples[key]["tcp_pose_mm_deg"][2] for key in ("z_min",)}
    WorkspaceBounds.from_mapping(updates)
    print(f"最低高度：{updates['z_min']:.2f} mm；无高度上限")
    if input("输入 SAVE 保存 Z 下限，其他输入取消：").strip() != "SAVE":
        print("已取消，未修改任何边界文件。")
        return
    save_component(args, updates, samples, remove=("z_max",))


def main():
    record_z(build_parser(__doc__).parse_args())


if __name__ == "__main__":
    main()
