"""Wait for this experiment's completed result and render deterministic reports."""
import time,subprocess
from pathlib import Path
ROOT=Path('/mnt/newssd/sja/clothAgent_real');R=ROOT/'reports/autonomous_retrieval_ab_20261102';E=ROOT/'experiments/autonomous_retrieval_ab_v1'
while not (R/'result.json').exists():time.sleep(5)
for python,script in [('/home/sja/miniconda3/envs/cali/bin/python','analyze.py'),('/usr/bin/python3','report.py'),('/home/sja/miniconda3/envs/cali/bin/python','validate.py')]:
 subprocess.run([python,str(E/script)],check=True,cwd=ROOT)
print('REPORT AND VALIDATION COMPLETE',flush=True)
