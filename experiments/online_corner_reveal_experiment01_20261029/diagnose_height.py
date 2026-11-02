"""Offline kinematic witness search; never connects to robot."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import differential_evolution
from scene_runtime import CheckedRuntime,ROOT
from cloth_agent.dual_arm.stop_sweep import native_joint_ranges
R=ROOT/'reports/online_corner_reveal_experiment01_20261029'
x=json.loads((R/'planned_action_04_tool_boundary/result.json').read_text())
e=json.loads(x['error'].split('stopping sweep UNKNOWN: ',1)[1]);seg=e['segment']
r=CheckedRuntime(ROOT/'config/dual_arm.local.json')
a={k:np.array(v) for k,v in seg['start_deg'].items()};b={k:np.array(v) for k,v in seg['end_deg'].items()}
lo,hi=native_joint_ranges(r.config,a,b,for_collision=True)
def height(q):
 c=next(c for c in r.models['left'].capsules(q) if c.name=='tool')
 return float(min(c.start[2],c.end[2])-c.radius)
normal=[height(a['left']+(b['left']-a['left'])*t) for t in np.linspace(0,1,101)]
sol=differential_evolution(height,list(zip(lo['left'],hi['left'])),seed=17,popsize=12,maxiter=250,tol=1e-9,polish=True)
w={'left':sol.x,'right':a['right']}
assert np.all(sol.x>=lo['left']-1e-9) and np.all(sol.x<=hi['left']+1e-9)
result={'normal_segment_sampled_min_tool_bottom_z_mm':min(normal),'normal_start_bottom_z_mm':height(a['left']),'normal_end_bottom_z_mm':height(b['left']),'stop_box_witness_bottom_z_mm':height(sol.x),'witness_joints_deg':sol.x.tolist(),'lower_joints_deg':lo['left'].tolist(),'upper_joints_deg':hi['left'].tolist(),'physical_table_z_mm':0,'duplicate_base_margin_applied_mm':0,'conclusion':'NO: a configuration inside existing stop bounds places modeled tool below Z0' if height(sol.x)<0 else 'NO: search did not find penetration; absence of a witness is not a full proof','optimizer_success':bool(sol.success),'scope':'Kinematic modeled witness, not observed physical motion; sampled nominal path minimum is not a continuous-path certificate.','robot_connections':0}
(R/'height_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
