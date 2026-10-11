"""Dual-arm folding using only the calibrated arm7 Camera B observation."""
import json
from pathlib import Path

from .codex_native_runtime import NativeRuntime
from .codex_motion import CONTRACT
from .dual_arm.geometry import DualArmError
from .codex_grasp_task import TASK_CONTRACT,lower_task_grasps


class DualArmRuntime(NativeRuntime):
    motion_contract=TASK_CONTRACT+(
        '\n以上宿主辅助动作及 grasp 格式限制仅适用于 grasp 模式。'
        'grasp 可仅含 left、仅含 right 或同时含两臂，各自目标可不同。'
        '也可选择以下显式模式，自行安排单臂/双臂移动、先后抓取、开合爪和释放；'
        '显式模式不写 grasp，不与 grasp 模式混用，结束前自行放稳并松开衣物。\n'+CONTRACT)

    def __init__(self,config_path):
        super().__init__(config_path)
        for model in self.models.values():
            model.ik_max_iterations=40
            model.ik_seed_limit=2

    def plan(self,source,initial,**kwargs):
        self._ik_exact_cache={}
        return super().plan(source,initial,**kwargs)

    def _inverse_target(self,arm,target_base,seed,phase):
        import numpy as np
        key=(arm,tuple(np.asarray(target_base,dtype=float)),tuple(np.asarray(seed,dtype=float)))
        cache=getattr(self,'_ik_exact_cache',{})
        if key in cache:
            self.plan_stats['ik_cache_hits']=self.plan_stats.get('ik_cache_hits',0)+1
            return cache[key].copy()
        q=super()._inverse_target(arm,target_base,seed,phase)
        if len(cache)<1024:cache[key]=q.copy()
        self._ik_exact_cache=cache
        return q

    preparation_description=(
        '采集只使用 arm7 的 Camera B，不启动 arm6 的 Camera A。'
        '两臂已在准备终点时直接采集；否则打开两爪、回 Home，再让 arm7 到 perception。'
        'grasp 模式由宿主负责最后松开与回 Home；显式模式自行安排释放，下一轮宿主准备拍照。\n')

    @property
    def planning_policy(self):
        from .codex_loop import COOPERATION_POLICY
        return COOPERATION_POLICY+(
            '两臂所有抓点均来自本轮 Camera B 的 world_xyz_mm.npy；不读取 Camera A。'
            '自行选择一臂或两臂、不同抓点和动作顺序，不要求两爪同时抓取或抓同一位置。grasp 模式由宿主规划辅助动作，显式模式按输出代码执行。'
            '宿主将 grasp 或显式 close 前的抓取 TCP 目标额外降低 10 mm。'
            '请输出观测原始抓取高度，不要重复减去该偏移。\n')

    def load_config(self,config_path):
        from .codex_camera_b import with_arm7_camera
        config=with_arm7_camera(super().load_config(config_path))
        config.raw['astra_grasp_height']={'mode':'independent_grasp_lowering'}
        return config

    def adjust_motion(self,source,folder):
        task=lower_task_grasps(source,folder)
        if task is not None:return task
        from .codex_grasp_height import lower_grasp_z
        changes=[]
        for arm in ('left','right'):
            source,rows=lower_grasp_z(source,arm=arm,offset_mm=10.)
            changes.extend(rows)
        if changes:
            (Path(folder)/'effective_motion.py').write_text(source)
            (Path(folder)/'grasp_height_policy.json').write_text(json.dumps(
                dict(mode='dual_grasp_lowering',offset_mm=10.,changes=changes),indent=2)+'\n')
        return source
