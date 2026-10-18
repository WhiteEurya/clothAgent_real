"""Queue certified position segments without changing their controller path."""
from dataclasses import dataclass

import numpy as np

from .dual_arm.geometry import DualArmError
from .dual_arm.sequential_home import PositionSegment,check_feedback
from .dual_arm.stop_sweep import native_joint_ranges


@dataclass
class QueuedPositionSegment(PositionSegment):
    segments: tuple

    def to_dict(self):
        return dict(super().to_dict(),queued_segments=[s.to_dict() for s in self.segments],
                    interpolation='unblended_controller_positions')


@dataclass
class CartesianPositionSegment(QueuedPositionSegment):
    target_pose: list
    cartesian_speed: float
    cartesian_acceleration: float

    def to_dict(self):
        return dict(super().to_dict(), interpolation='controller_linear_tcp',
                    target_pose_mm_deg=self.target_pose,
                    cartesian_speed_mm_s=self.cartesian_speed,
                    cartesian_acceleration_mm_s2=self.cartesian_acceleration)


def queue_moves(operations,limit=32):
    """Preserve all waypoints and gripper boundaries; no geometric smoothing."""
    result=[];pending=[]
    def flush():
        if len(pending)==1:result.append(('move',pending[0]))
        elif pending:
            first,last=pending[0],pending[-1]
            result.append(('move',QueuedPositionSegment(first.arm,first.start,last.end,
                min(s.speed for s in pending),min(s.acceleration for s in pending),
                sum(s.timeout for s in pending),tuple(pending))))
        pending.clear()
    for kind,data in operations:
        if pending and (kind!='move' or data.arm!=pending[-1].arm or len(pending)>=limit):flush()
        if kind=='move':
            if pending and any(not np.allclose(pending[-1].end[k],data.start[k],atol=1e-9,rtol=0) for k in data.start):
                raise DualArmError('queued path is discontinuous')
            pending.append(data)
        else:result.append((kind,data))
    flush()
    return result


def check_queued_feedback(coordinator,batch):
    state=coordinator.snapshots()
    joints={k:np.asarray(v['joints']) for k,v in state.items()}
    for segment in batch.segments:
        lo,hi=native_joint_ranges(coordinator.config,segment.start,segment.end)
        if all(np.all(joints[k]>=lo[k]-1e-9) and np.all(joints[k]<=hi[k]+1e-9) for k in joints):
            # Reuse the same snapshot: near a waypoint a second read could
            # already belong to the next segment.
            from types import SimpleNamespace
            check_feedback(SimpleNamespace(config=coordinator.config,models=coordinator.models,
                           snapshots=lambda:state,event=coordinator.event),segment)
            return
    raise DualArmError('controller left certified queued path')
