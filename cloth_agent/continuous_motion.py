"""Offline continuous-speed candidates that stay in the original joint boxes.

These artifacts are not executable commands: servo commissioning and an
execution adapter are required before replacing native position execution.
"""
import numpy as np

from .dual_arm.geometry import DualArmError
from .dual_arm.motion.home import pack
from .dual_arm.motion.trajectory import parameterize,polynomial_peak,sample_segment


def compile_continuous(operations):
    groups=[];pending=[]
    def flush():
        if pending:
            groups.append(_group(pending));pending.clear()
    for row in operations:
        if row['kind']!='move':
            flush();groups.append(row);continue
        data=row['data']
        for segment in data.get('queued_segments',[data]):
            if pending and segment['arm']!=pending[-1]['arm']:flush()
            pending.append(segment)
    flush()
    return dict(schema_version=1,execution_authorized=False,
                required_execution_mode='commissioned_servo_stream',groups=groups)


def _group(moves):
    arm=moves[0]['arm'];edges=[]
    for move in moves:
        start,end=pack(move['start']),pack(move['end'])
        idle=slice(6,13) if arm=='left' else slice(0,6)
        if not np.allclose(start[idle],end[idle],atol=1e-12,rtol=0):
            raise DualArmError('continuous candidate must keep inactive arm stationary')
        edges.append(dict(phase='continuous',start_q_rad=start.tolist(),end_q_rad=end.tolist()))
    v=np.full(13,np.deg2rad(min(m['speed_deg_s'] for m in moves)))
    a=np.full(13,np.deg2rad(min(m['acceleration_deg_s2'] for m in moves)))
    trajectory=parameterize(edges,v,a)
    original=trajectory['segments']
    # Shared knot velocities keep C2 continuity. Reduce them together if the
    # quintic rounds outside any already-certified endpoint joint box.
    selected=None
    for factor in [1.,.5,.25,.125,.0625,.03125,0.]:
        rows=[];valid=True;scale=1.
        for segment in original:
            duration=segment['duration_s']
            first,last=sample_segment(segment,0),sample_segment(segment,duration)
            start,end=first[0],last[0];d=end-start
            v0,v1=first[1]*duration*factor,last[1]*duration*factor
            c=np.array([start,v0,np.zeros(13),10*d-6*v0-4*v1,
                        -15*d+8*v0+7*v1,6*d-3*v0-3*v1])
            centered=c.copy();centered[0]-=(start+end)/2
            if np.any(polynomial_peak(centered)>np.abs(d)/2+1e-12):
                valid=False;break
            speed=polynomial_peak(np.polynomial.polynomial.polyder(c))/duration
            accel=polynomial_peak(np.polynomial.polynomial.polyder(c,m=2))/duration**2
            scale=max(scale,float(np.max(speed/v)),float(np.sqrt(np.max(accel/a))))
            rows.append(dict(segment,coefficients_rad=c.tolist()))
        if valid:selected=rows;break
    if selected is None:raise DualArmError('cannot contain continuous candidate in certified joint boxes')
    clock=0.
    for row in selected:
        row['duration_s']*=scale*(1+1e-10);row['start_time_s']=clock;clock+=row['duration_s']
    speeds=[float(np.max(np.abs(sample_segment(s,s['duration_s'])[1]))) for s in selected[:-1]]
    return dict(kind='continuous_candidate',arm=arm,segments=selected,duration_s=clock,
                contained_in_original_joint_boxes=True,knot_velocity_scale=factor,
                moving_internal_knots=sum(s>1e-8 for s in speeds),internal_knots=len(speeds),
                velocity_limits_rad_s=v.tolist(),acceleration_limits_rad_s2=a.tolist())
