---
name: cloth-contour-normal-grasp
description: 要展开的对应衣角或袖口被其他布层覆盖、不可见时，定位其附近的可见轮廓，由局部法线程序生成对侧抓放动作。
---

Simple只能抓四个边角（两个下摆角和两个袖口）。仅当要展开的对应衣角或袖口被其他布层覆盖、不可见时使用几何 Skill；对应边角可见时用 Simple。一次未揭开且该边角仍被覆盖时可继续使用 Skill；不能仅因大块内收、褶皱、歪斜或角点难辨触发。
- 输入：在该部位附近的可见外轮廓选定位点 Q，返回 boundary_point_px=Q、chord_endpoints_px=[]。不为得到预设抓点反复挑选 Q。不要猜隐藏角坐标。附 region_name、evidence 和期望水平拉距 pull_distance_mm（正数、毫米，不设固定上下限；须满足工作边界和完整路径检查）。
- 程序：返回 skill="cloth-contour-normal-grasp"，grasp_px/target_px=null；宿主调用现有 measured_candidate → contour_normal_grasp.geometry。CV 计算内法线、正向最远交点和向外抓放点，模型不重算。
- 空间不足：原定对向拉动会出界时，可先用 make-space-for-skill 反方向抓放留出空间；同一 round 中重新拍照后，再按新轮廓规划正向揭开。两段分别检查，第一段失败不继续；第二段不再追加腾空间动作。
- 结果：候选沿既有深度、分区、运动检查执行；无法生成时使用宿主现有失败或重选流程。

完整原理及历史示例见 [参考文件](references/full_method.md)，仅开发排查时阅读，正常单次规划无需加载。
