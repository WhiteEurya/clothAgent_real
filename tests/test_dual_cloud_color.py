import numpy as np
from cloth_agent.dual_cloud_color import match_colors


def test_stale_table_transform_cannot_recolor_black_cloth():
    rgb=np.full((100,100,3),180,np.uint8);rgb[20:80,20:80]=20
    y,x=np.indices((100,100));xyz=np.stack([x,y,np.zeros_like(x)],axis=2).astype(float)
    gain,offset,report=match_colors([rgb,rgb],[xyz,xyz],{'gain_rgb':[.64,.95,1.45],'offset_rgb':[80,37,-33]})
    assert np.array_equal(gain,np.ones(3)) and np.array_equal(offset,np.zeros(3))
    assert report['dark_after']>report['dark_before']
    assert report['status']=='IDENTITY_UNVALIDATED_COLOR'
