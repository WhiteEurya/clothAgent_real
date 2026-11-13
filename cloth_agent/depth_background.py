"""Offline per-camera native-depth background subtraction; no robot access."""
import warnings
import numpy as np


def fit_background(frames_m, *, min_valid_fraction=.8):
    frames=np.array(frames_m,dtype=np.float32,copy=True)
    if frames.ndim!=3 or frames.shape[0]<10:
        raise ValueError('background requires at least 10 native depth frames')
    valid=np.isfinite(frames)&(frames>0)
    frames[~valid]=np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(frames,axis=0)
        sigma=1.4826*np.nanmedian(np.abs(frames-median),axis=0)
    support=valid.mean(0)>=min_valid_fraction
    return dict(median_m=median,sigma_m=sigma,supported=support,
                valid_count=valid.sum(0),frame_count=np.array(len(frames)))


def subtract_background(depth_m, background, *, minimum_change_mm=5., noise_multiplier=3.):
    depth=np.asarray(depth_m)
    if depth.shape!=background['median_m'].shape:raise ValueError('native depth grid changed')
    if minimum_change_mm<=0 or noise_multiplier<0:raise ValueError('invalid depth difference threshold')
    known=np.isfinite(depth)&(depth>0)&background['supported']
    delta=(background['median_m']-depth)*1000.
    threshold=minimum_change_mm+noise_multiplier*background['sigma_m']*1000.
    foreground=known&(delta>threshold)
    # Missing depth remains unknown, never inferred as cloth.
    return foreground,known,delta,threshold
