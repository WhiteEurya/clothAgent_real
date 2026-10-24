import pytest

from cloth_agent.experiments.dino_action_grounding import action_points, validate


def test_original_image_bounds_and_abstention():
    validate(dict(status='PROPOSED', grasp_xy=[1,2], target_xy=[10,20]), (1280,720))
    validate(dict(status='NOT_APPLICABLE', grasp_xy=None, target_xy=None), (1280,720))
    for status,a,b in [('PROPOSED',None,None), ('PROPOSED',[1280,2],[5,6]),
                       ('PROPOSED',[1,2],[1,2]), ('UNCERTAIN',[1,2],None),
                       ('NOT_APPLICABLE',[1,2],[5,6])]:
        with pytest.raises(ValueError):validate(dict(status=status,grasp_xy=a,target_xy=b),(1280,720))


def test_missing_point_pair_does_not_substitute_a_different_reference():
    candidate={'edge':{'seed_action':{'exact_command':{'selected_points':[]}}}}
    with pytest.raises(ValueError, match='do not substitute'):action_points(candidate)
    candidate['edge']['seed_action']['exact_command']['selected_points']=[
        dict(name='target',pixel_xy=[30,40]),dict(name='start',pixel_xy=[10,20])]
    assert action_points(candidate)==([10,20],[30,40])
