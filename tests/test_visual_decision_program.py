import copy
import pytest
from cloth_agent.experiments.visual_decision_program import decide,validate_program,tree

@pytest.fixture
def program():
    return dict(regions=[dict(id='upper',semantic_region='upper fabric')],
        predicates=[dict(id=i,region='upper',question='Is the boundary visible?') for i in ('a','b')],
        rules=[{'if':'a == YES && b == NO','result':'S1'}, {'if':'b == YES','result':'S2'}],fallback='UNKNOWN')

def values(a,b):return [dict(predicate_id='a',value=a),dict(predicate_id='b',value=b)]

@pytest.mark.parametrize('a,b,result', [('YES','NO','S1'),('NO','YES','S2'),('NO','NO','UNKNOWN'),
                                     ('UNKNOWN','YES','UNKNOWN'),('YES','UNKNOWN','UNKNOWN')])
def test_host_rules_and_required_unknown(program,a,b,result):
    assert decide(program,values(a,b))['decision']==result

def test_conflict_returns_unknown(program):
    program['rules'][0]['if']='a == YES || b == NO'
    assert decide(program,values('YES','YES'))['reason']=='conflict'

@pytest.mark.parametrize('expr',["__import__('os').system('id')",'a != YES','c == NO','a == UNKNOWN','a','a == YES + NO'])
def test_rules_are_not_python_execution(expr):
    with pytest.raises((ValueError,SyntaxError)):tree(expr,{'a','b'})

def test_reject_extra_missing_duplicate_results(program):
    for v in [values('YES','NO')[:1],values('YES','NO')*2,[dict(predicate_id='x',value='NO'),dict(predicate_id='b',value='NO')]]:
        with pytest.raises(ValueError):decide(program,v)

def test_immutable_input_and_unknown_region(program):
    before=copy.deepcopy(program);decide(program,values('YES','NO'));assert program==before
    program['predicates'][0]['region']='absent'
    with pytest.raises(ValueError):validate_program(program)
