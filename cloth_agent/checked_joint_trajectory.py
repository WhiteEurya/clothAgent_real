"""Time-parameterized joint curves contained in existing checked joint boxes."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from .config import SafetyError
from .dual_arm.motion.trajectory import polynomial_peak


def compile_trajectory(points, speed, acceleration, period=.01, *, tcp_reaches=None, tcp_speed=None, tcp_acceleration=None):
    q=np.asarray(points,dtype=float)
    if q.ndim!=2 or len(q)<2 or not np.isfinite(q).all():
        raise SafetyError('invalid continuous joint path')
    if not np.isfinite([speed,acceleration,period]).all() or min(speed,acceleration)<=0 or period!=.01:
        raise SafetyError('invalid continuous joint timing')
    q=q[np.r_[True,np.max(np.abs(np.diff(q,axis=0)),axis=1)>1e-9]]
    if len(q)<2:raise SafetyError('continuous joint path has no movement')
    delta=np.diff(q,axis=0)
    distance=np.max(np.abs(delta),axis=1)
    durations=np.maximum(period,np.maximum(distance/speed, np.sqrt(6*distance/acceleration)))
    def fit(timing):
        knots=np.r_[0.,np.cumsum(timing)]
        slopes=PchipInterpolator(knots,q,axis=0).derivative()(knots)
        slopes[[0,-1]]=0
        for factor in (1.,.5,.25,.125,.0625,0.):
            rows=[];scales=[]
            for i,(d,t) in enumerate(zip(delta,timing)):
                v0,v1=slopes[i]*t*factor,slopes[i+1]*t*factor
                c=np.array([q[i],v0,np.zeros(q.shape[1]),10*d-6*v0-4*v1,
                            -15*d+8*v0+7*v1,6*d-3*v0-3*v1])
                centered=c.copy();centered[0]-=(q[i]+q[i+1])/2
                if np.any(polynomial_peak(centered)>np.abs(d)/2+1e-9):break
                scales.append(max(float(np.max(polynomial_peak(np.polynomial.polynomial.polyder(c))/t/speed)),
                                  float(np.sqrt(np.max(polynomial_peak(np.polynomial.polynomial.polyder(c,2))/t**2/acceleration)))))
                rows.append(c)
            else:return rows,np.asarray(scales),slopes,factor
        raise SafetyError('continuous curve leaves checked joint boxes')

    # Recompute shared velocities after every local timing adjustment. Only the
    # final analytically checked candidate is executable; iterations are offline.
    best=None
    for iteration in range(13):
        rows,scales,slopes,factor=fit(durations)
        safe_durations=durations*max(float(np.max(scales)),1e-9)*(1+1e-8)
        if best is None or safe_durations.sum()<best[0].sum():
            best=(safe_durations,rows,slopes,factor)
        durations=np.maximum(period,durations*np.clip(scales,.65,1.5))
    durations,selected,slopes,factor=best
    tcp_scale=1.
    if tcp_speed is not None or tcp_acceleration is not None:
        weights=np.asarray(tcp_reaches,float)
        if (weights.shape!=(q.shape[1],) or not np.isfinite(weights).all() or np.any(weights<0)
                or not np.any(weights>0) or tcp_speed is None or tcp_acceleration is None
                or not np.isfinite([tcp_speed,tcp_acceleration]).all() or min(tcp_speed,tcp_acceleration)<=0):
            raise SafetyError('invalid TCP timing limits')
        # Serial-chain reach bounds cover all joint configurations, including
        # intermediate polynomial poses. Uniform time stretching preserves
        # the exact checked geometric curve and scales v by 1/k, a by 1/k².
        for c,t in zip(selected,durations):
            omega=np.radians(polynomial_peak(np.polynomial.polynomial.polyder(c))/t)
            alpha=np.radians(polynomial_peak(np.polynomial.polynomial.polyder(c,2))/t**2)
            v_bound=float(weights@omega)
            a_bound=float(weights@alpha+max(weights)*omega.sum()**2)
            tcp_scale=max(tcp_scale,v_bound/tcp_speed,np.sqrt(a_bound/tcp_acceleration))
        durations=durations*tcp_scale*(1+1e-8)
    ends=np.cumsum(durations);total=float(ends[-1]);times=np.r_[np.arange(0,total,period),total]
    if len(times)>200000:raise SafetyError('continuous trajectory sample budget exceeded')
    starts=np.r_[0.,ends[:-1]];samples=[]
    for t in times:
        i=min(int(np.searchsorted(ends,t,side='right')),len(ends)-1)
        samples.append(np.polynomial.polynomial.polyval((t-starts[i])/durations[i],selected[i]).tolist())
    return dict(tcp_time_scale=tcp_scale,tcp_speed_limit_mm_s=tcp_speed,tcp_acceleration_limit_mm_s2=tcp_acceleration,duration_s=total,period_s=period,times_s=times.tolist(),positions_deg=samples,
                coefficients= [c.tolist() for c in selected],segment_durations_s=durations.tolist(),
                knot_velocity_factor=factor,moving_internal_knots=int(np.sum(np.max(np.abs(slopes[1:-1]*factor),axis=1)>1e-8)),
                contained_in_checked_joint_boxes=True,speed_deg_s=speed,acceleration_deg_s2=acceleration)
