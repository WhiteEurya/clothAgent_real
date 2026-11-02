---
name: cloth-contour-normal-grasp
description: 利用 CV 分割衣物轮廓，对视觉判断为不可见或内收的区域，在其邻近可见轮廓作法线，求对侧交点并生成一次向外拉动动作。用于衣物展开中的轮廓法线抓取、离线选点及经授权的单次实机执行；不自动认定隐藏衣角，也不使用 HEM_REVEAL。
---

# 衣物轮廓法线抓取

在 clothAgent_real 项目中运行，默认 Python 为 `/home/sja/miniconda3/envs/cali/bin/python`。实现位于 `cloth_agent/contour_normal_grasp.py`，单次执行入口位于 `cloth_agent/experiments/contour_normal_grasp_once.py`。如当前目录不是该项目，先定位项目根目录；不要把其他仓库当成兼容环境。

## 输入和方向

先查看本轮 RGB。指定不可见区域的名称、可见依据，以及靠近该区域的**可见外轮廓点** `Q` 的像素位置。该位置用来定位局部轮廓，不代表隐藏衣角坐标。不可见区域的语义判断由用户或 Agent 提供，CV 不代替这个判断。

本 skill 的方向已经确定：从 Q 沿轮廓法线进入衣服，穿至对侧交点 E，在 E 附近抓取，再沿同一方向继续向外拉。不是从交点朝 Q 拉，也不是把布翻回衣身中心。

无法关联到一段可见轮廓时，报告信息不足，不补画一条虚构的隐藏轮廓。局部轮廓点不同可能得到不同交点，记录选点依据，不为得到预设抓点反复挑选 Q。

## 生成并检查候选

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.contour_normal_grasp \
  --image /absolute/path/rgb.png \
  --region-point X Y \
  --region-name '不可见区域名称' \
  --evidence '当前图中支持该判断的可见依据' \
  --output reports/unique_normal_preview
```

必须替换 X、Y 和路径。输出目录需尚不存在。

- 默认分割复用当前实验的“紫色衣服／深色桌面／固定侧边 ROI”算法，不是通用衣物分割器。其他衣物或相机视角用 `--mask /path/visible_mask.png` 提供与原图逐像素对齐的二值可见布料掩膜。
- `--tangent-window-px 25` 控制局部轮廓 PCA 拟合窗口；轮廓切线的垂线就是法线。
- 交点 E 保存在 `proposal.json`。实际 Pa 默认沿法线退入布内 8 px（`--inset-px`），且 5×5 抓点邻域必须全部属于分割出的布料。保留 E 与 Pa 的区别。
- `--pull-px 80` 是默认的**像素候选幅度**，不是毫米或测量结果。可以根据动作目的指定幅度；最终必须由 RGB-D 和 Host 校验 10–150 mm 范围，不通过则停止。
- 输出 `mask.png`、含完整轮廓／法线／交点的 `proposal.json`、`plan.json`、`action.png` 和 `result.json`。
- 若分割截断、多次穿出轮廓、切线不稳定、交点邻域无足够可见布料或 Pb 未落到轮廓外，返回 `ABSTAIN`，不会连接机械臂。

查看 `mask.png` 与 `action.png`。检查掩膜是否混入桌边／人／夹爪，Q 是否对应所选区域，交点是否确为可见可抓布料。法线可能交到领口或其他部位，不能因几何计算成功就把交点叫成袖口或下摆角；若本次实验另有限定抓取部位，仍须满足该限定。几何候选不能证明展开效果。

最终向用户显示 Q、E、Pa、Pb、方向和预期作用，并区分已完成的几何计算与尚未验证的衣物动力学。不要让用户以为已经执行。

## 经授权执行一次

用户只要求创建 skill、看图或规划时不执行实机。已有明确单次执行授权时直接使用，不重复索要确认。

必须有对应的真实 `Observation` 文件夹（RGB、观测元数据、对齐 XYZ），其 RGB 哈希须与 proposal 一致。单张上传图片只支持预览，不能配用其他时刻的深度。

```bash
/home/sja/miniconda3/envs/cali/bin/python -m cloth_agent.experiments.contour_normal_grasp_once \
  --observation /absolute/path/observation \
  --proposal reports/unique_normal_preview/proposal.json \
  --output reports/unique_normal_execution \
  --execute
```

入口复用 Arm6 `EpisodeCollectionRuntime`：重新观察、检查场景变化，再走 `prepare_action` 的实测抓点深度、桌面目标投影、距离、工作空间和 IK 等现有配置下的检查，然后只派发一次 `lift_pull_place`。它不启用原运行时未启用的检查，也不修改限速或检查阈值。抓前沿用现有 Home 姿态，持布不旋转；抬升和释放使用当前普通动作参数。

返回失败或抛出异常即停止，不自动重试。`dispatch_intent.json` 标记同一执行输出已派发，禁止复用。成功执行后回到观测位拍摄 after；`EXECUTED` 仅表示命令执行，仍需看 before/after 判断不可见区域是否展开。

已有运行时可直接调用 `execute_once(runtime, observation, proposal, output)`，不另开第二个控制连接。
