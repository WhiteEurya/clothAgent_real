"""Run supplemental matched-information and effort controls after the main suite."""
import argparse,json,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--iteration',type=Path,required=True);a=p.parse_args()
start=time.monotonic()
while True:
    path=a.output/'results.json'
    rows=json.loads(path.read_text()) if path.exists() else []
    if len(rows)>=8:break
    if time.monotonic()-start>14400:raise TimeoutError('Main suite did not complete')
    time.sleep(5)
for arm,extra in [('H_low_effort',['--effort','low']),('I_matched_inline',['--matched-run',str(a.output)])]:
    dest=a.output/('supplement_'+arm[0]);print('START',arm,flush=True)
    cmd=[sys.executable,'scripts/benchmark_planner_latency.py','--iteration',str(a.iteration),'--output',str(dest),'--arms',arm,*extra]
    subprocess.run(cmd,check=True)
    rows=json.loads((a.output/'results.json').read_text())
    rows+=json.loads((dest/'results.json').read_text())
    (a.output/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print('ALL_SUPPLEMENTS_FINISHED',flush=True)
