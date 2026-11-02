"""Wait for frozen analysis, then run authorized post-freeze diagnostic and rendering."""
import json,time,subprocess,sys
from pathlib import Path
base=Path('/mnt/newssd/sja/clothAgent_real');root=base/'reports/hypothesis_retrieval_20261102'
while not (root/'analysis_frozen.json').exists():time.sleep(10)
with (root/'report_build.log').open('w') as log:
 for cmd in [[str(Path('/home/sja/miniconda3/envs/cali/bin/python')),str(base/'experiments/hypothesis_retrieval_20261102/run.py'),'diagnostic','--experience-id','X0518'],['/usr/bin/python3',str(base/'experiments/hypothesis_retrieval_20261102/report.py')]]:
  r=subprocess.run(cmd,stdout=log,stderr=log)
  if r.returncode:raise SystemExit(r.returncode)
(root/'delivery_complete.json').write_text(json.dumps({'status':'COMPLETE','time':time.time(),'files':['review.html','retrieval_report.pdf','retrieval_full_index.pdf','retrieval_analysis.json']},indent=2))
print('Reports ready',flush=True)
