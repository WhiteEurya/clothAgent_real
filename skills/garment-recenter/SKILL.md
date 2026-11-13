---
name: garment-recenter
description: 桌边限制继续展开时，抓衣领附近布料，按衣物中心到桌面中心的位移恢复操作空间。
---

若桌边挡住所需展开方向（包括无法继续向下拉），先恢复空间，不要仅为避开桌沿持续改向侧拉；Simple 与几何 Skill 都适用。已展开可完成则结束，不为精确居中继续操作。

在当前 topdown 中选择衣领附近清楚可见、可夹住的布料，避开领口孔洞，输出 skill="garment-recenter" 和 grasp_px。target_px、boundary_point_px、grasp_angle_deg=null，chord_endpoints_px=[]；无法定位可靠抓点则 ABSTAIN。不额外调用模型确认衣领。

Host 用现有 mask 计算衣物面积质心 C，用标定桌面区域计算中心 T，将 D=T−C 应用到所选抓点 G，放点为 P=G+D；不是把抓点放到桌面中心。当前单步沿用150mm限制，较长位移先执行不超过140mm的一段，再拍照重新计算。pull_distance_mm 不决定恢复距离。

抓住后先垂直抬到抓放两点较高实测表面上方50mm，再水平平移并放下；不是绝对world Z=50mm。

复用现有实测深度、同臂完整路径、IK、关节及碰撞检查，动作后重新观察；不把计划位移当作布料整体实际位移。沿用普通执行速度。独立离线位移计算见 [scripts/plan.py](scripts/plan.py)，不连接机器人。
