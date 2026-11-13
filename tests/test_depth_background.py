import numpy as np
import pytest
from cloth_agent.depth_background import fit_background,subtract_background


def test_nonplanar_table_cancels_and_raised_cloth_is_detected():
    y,x=np.indices((30,40));table=.8+.02*np.sin(x/8)+.003*y
    frames=np.repeat(table[None],20,axis=0)
    model=fit_background(frames);current=table.copy();current[8:20,10:30]-=.025
    original=current.copy();mask,known,_,_=subtract_background(current,model)
    expected=np.zeros(table.shape,bool);expected[8:20,10:30]=True
    assert np.array_equal(mask,expected) and known.all()
    assert np.array_equal(current,original)


def test_missing_background_and_current_depth_stay_unknown():
    frames=np.full((20,10,10),1.);frames[:,:,0]=0;frames[:,0,:]=np.nan
    model=fit_background(frames);current=np.full((10,10),.9);current[:,2]=0
    mask,known,_,_=subtract_background(current,model)
    assert not known[:,0].any() and not known[0].any() and not known[:,2].any()
    assert not mask[~known].any()


def test_local_noise_changes_threshold_not_global_height():
    frames=np.ones((20,10,10));frames[::2,:,5:]+=.01;frames[1::2,:,5:]-=.01
    model=fit_background(frames);mask,_,_,threshold=subtract_background(np.full((10,10),.99),model)
    assert mask[:,:5].all() and not mask[:,5:].any()
    assert threshold[:,5:].min()>threshold[:,:5].max()
    with pytest.raises(ValueError):subtract_background(np.ones((5,5)),model)
