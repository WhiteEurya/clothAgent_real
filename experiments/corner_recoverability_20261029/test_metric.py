from metric import *
A=[0,1,1,1];B=[1,0,1,1];C=[1,1,1,1]
def test_required_chain():
 s=analyze([A,B,C]);assert [x['N']['exact'] for x in s]==[3,3,4]
 assert [x['R2']['value'] for x in s]==[.81,.9,1.]
 assert s[0]['nonincrease_then_observed_goal'] and s[0]['exchange_next']
def test_cycle_and_censor():
 s=analyze([A,B,A]);assert s[0]['R2']['value']==0
 assert s[1]['R2']['status']=='CENSORED' and s[1]['R2']['value'] is None
 assert returns([A,B],0)['value'] is None
def test_unknown_first_hit():
 u=[None,1,1,1];r=returns([A,u,C],0)
 assert r['status']=='UNKNOWN' and r['lower']==.81 and r['upper']==.9
 assert count(u)=={'exact':None,'lower':3,'upper':4}
def test_known_false_with_unknown_corner():
 assert goal([0,None,1,1])==0
 assert returns([[0,None,1,1],B,C],0)['value']==.81
def test_goal_current_and_horizon():
 assert returns([C],0)['value']==1
 assert returns([A,B,A,C],0)['value']==0
 assert returns([A,B,A,C],1)['value']==.81
def test_unknown_is_not_zero():
 assert returns([[None]*4]*3,0)['status']=='UNKNOWN'
 assert returns([A,[None]*4],0)['censored']
