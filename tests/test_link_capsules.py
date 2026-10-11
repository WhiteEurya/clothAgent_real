from itertools import product
import numpy as np
import pytest
from cloth_agent.dual_arm.link_capsules import cover_box


def test_fixed_radius_capsules_cover_entire_expanded_box():
    vertices=np.array(list(product([-180.,180.],[-85.,85.],[-60.,60.])))
    caps,proof=cover_box(vertices,'link2',30.,shell_mm=2.)
    assert len(caps)>1 and caps[0]['name']=='link2'
    assert len({c['name'] for c in caps})==len(caps)
    assert proof['cell_corner_distance_mm']<=30
    assert all(c['radius_mm']==30 and c['frame']=='link2' for c in caps)
    # Arbitrary interior points are covered too, not just surface vertices.
    rng=np.random.default_rng(42)
    local=rng.uniform(proof['box_min'],proof['box_max'],size=(5000,3))
    points=np.array(proof['center'])+local@np.array(proof['basis'])
    distances=[]
    for c in caps:
        a,b=np.array(c['start_mm']),np.array(c['end_mm']);delta=b-a
        t=np.clip((points-a)@delta/(delta@delta),0,1)
        distances.append(np.linalg.norm(points-a-t[:,None]*delta,axis=1))
    assert np.max(np.min(distances,axis=0))<=30


@pytest.mark.parametrize('radius',[0,-1,float('nan')])
def test_invalid_radius_rejected(radius):
    with pytest.raises(ValueError):cover_box(np.eye(3),'link1',radius)


def test_many_capsule_broadphase_matches_brute_force_first_collision():
    from cloth_agent.dual_arm.geometry import Capsule,check_collision,segment_distance,DualArmError
    rng=np.random.default_rng(72)
    for offset in (0,100,1000):
        for _ in range(10):
            caps={k:[Capsule(f'c{i}',rng.normal(shift,30,3),rng.normal(shift,30,3),5.) for i in range(20)]
                  for k,shift in [('left',0),('right',offset)]}
            expected=next(((a.name,b.name) for a in caps['left'] for b in caps['right']
                           if segment_distance(a.start,a.end,b.start,b.end)<=a.radius+b.radius+2),None)
            if expected:
                with pytest.raises(DualArmError) as exc:check_collision(caps,[],2)
                assert str(exc.value)==f'inter-arm collision: left/{expected[0]}, right/{expected[1]}'
            else:check_collision(caps,[],2)
