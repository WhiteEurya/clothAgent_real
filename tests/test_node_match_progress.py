import pytest
from cloth_agent.experiments.node_match_progress import node_match_progress
from cloth_agent.experiments.edge_applicability import SCHEMAS

@pytest.mark.parametrize('result,reliable,conflict,label',[
 ('S4',True,False,'PROGRESS'),('S3',True,False,'NO_STATE_CHANGE'),
 ('S5',True,False,'ALTERNATIVE_OUTCOME'),(None,True,False,'UNKNOWN'),
 ('S4',False,False,'UNKNOWN'),('S4',True,True,'UNKNOWN')])
def test_matching_is_only_progress_authority(result,reliable,conflict,label):
    assert node_match_progress('S3','S4',result,reliable=reliable,conflict=conflict)['progress']==label


def test_role_schema_cannot_return_progress():
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(dict(action_relation='SAME_ROLE',progress='NO_PROGRESS',evidence=['x'],decision_condition='x'),SCHEMAS['applicability_outcome'])
