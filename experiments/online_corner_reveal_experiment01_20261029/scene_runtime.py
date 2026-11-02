"""Experiment-local scene correction: operator confirms Arm7 physically removed.
The paired interface retains an inert right-arm placeholder; no right connection.
Only geometry belonging to the absent assembly is removed. Arm6 geometry,
stop envelopes, table obstacles, IK and full path verification remain unchanged.
"""
from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'corner_reveal_physical_E353_20261029'))
from preflight import CheckedRuntime as OriginalCheckedRuntime, ROOT
class CheckedRuntime(OriginalCheckedRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        config=self.config
        before=copy.deepcopy(config.arms['left'].capsules)
        config.arms['right'].capsules=[]
        config.arms['right'].raw=copy.deepcopy(config.arms['right'].raw)
        config.arms['right'].raw['collision_capsules']=[]
        # Keep the paired numerical placeholder to avoid changing controller APIs.
        # Existing static connection cannot issue robot commands or open an IP.
        config.static_arm_obstacles['right'].update(assumption='physically_removed_operator_confirmed_20261029',physical_presence=False)
        assert config.arms['left'].capsules==before
        assert config.path_validation_enabled and not config.collision_envelopes_disabled_arms
        # Retain cached kinematic reach bounds for the inert paired interface.
        # Collision geometry is read from config.capsules and is now empty.
        # Operator-specified tool boundary already includes base calibration error.
        # Represent its non-duplicated constraint using a separate tool-only plane:
        # (nominal tool bottom - stop envelope - base_error) >= -14.9-base_error
        # iff (nominal tool bottom - stop envelope) >= physical Z=-14.9.
        # This is a collision-coordinate compensation, NOT a physical table shift; only the tool has an operator-confirmed
        # compression boundary at Z=-14.9. Other links retain Z=0.
        table=next(b for b in config.obstacles if b['name']=='table_z0_user_confirmed')
        base_error=float(config.raw['safety']['arms']['left']['base_error_mm'])
        assert table['max_mm'][2]==0 and base_error==10.
        tool_table=copy.deepcopy(table)
        tool_table['name']='tool_min_z_minus14p9_operator_confirmed'
        tool_table['max_mm'][2]=-14.9-base_error
        tool_table['excluded_capsules']=[f'{k}/{c["name"]}' for k,a in config.arms.items() for c in a.capsules if (k,c['name'])!=('left','tool')]
        tool_table['source']='Operator confirms physical lowest tool Z>=-14.9; base uncertainty already included. Compensates duplicate base margin only; stop envelope retained.'
        table.setdefault('excluded_capsules',[]).append('left/tool')
        config.obstacles.append(tool_table)
