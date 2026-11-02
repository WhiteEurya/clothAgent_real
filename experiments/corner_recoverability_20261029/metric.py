"""Historical first-hitting returns, never optimal policy values."""
PARTS=('LS','RS','LH','RH')
def goal(v):
 if len(v)!=4 or any(x not in (0,1,None) for x in v):raise ValueError('four tri-state labels required')
 if all(x==1 for x in v):return 1
 if any(x==0 for x in v):return 0
 return None

def count(v):
 goal(v);lo=sum(x==1 for x in v);hi=lo+sum(x is None for x in v)
 return {'exact':lo if lo==hi else None,'lower':lo,'upper':hi}

def returns(states,t,H=2,gamma=.9):
 if H<0 or not 0<gamma<=1 or not 0<=t<len(states):raise ValueError('invalid parameters')
 uncertain=[]
 for k in range(min(H,len(states)-1-t)+1):
  g=goal(states[t+k])
  if g is None:uncertain.append(k)
  elif g==1:
   if uncertain:return {'status':'UNKNOWN','value':None,'tau':None,'lower':gamma**k,'upper':gamma**uncertain[0],'reason':'possible earlier goal at uncertain observation','censored':False}
   return {'status':'OBSERVED_SUCCESS','value':gamma**k,'tau':k,'lower':gamma**k,'upper':gamma**k,'censored':False}
 complete=t+H<len(states)
 if uncertain:return {'status':'UNKNOWN','value':None,'tau':None,'lower':0.,'upper':gamma**uncertain[0],'reason':'uncertain goal observation','censored':not complete}
 if not complete:return {'status':'CENSORED','value':None,'tau':None,'lower':0.,'upper':gamma**(len(states)-t),'reason':'observed suffix shorter than horizon','censored':True}
 return {'status':'OBSERVED_NO_HIT','value':0.,'tau':None,'lower':0.,'upper':0.,'censored':False}

def analyze(states):
 result=[]
 for t,v in enumerate(states):
  n=count(v);r=returns(states,t);nxt=count(states[t+1]) if t+1<len(states) else None
  change=nxt['exact']-n['exact'] if nxt and nxt['exact'] is not None and n['exact'] is not None else None
  exchange=bool(nxt and any(a==0 and b==1 for a,b in zip(v,states[t+1])) and any(a==1 and b==0 for a,b in zip(v,states[t+1])))
  result.append({'t':t,'visibility':v,'G':goal(v),'N':n,'R2':r,'delta_N_next':change,'exchange_next':exchange,'nonincrease_then_observed_goal':change is not None and change<=0 and r['status']=='OBSERVED_SUCCESS' and r['tau']>0})
 return result
