"""Replay the current segmentation entrypoint on saved observations; no hardware."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.experiments.unfold_dual_topdown import reconstruct
from cloth_agent.experiments.unfold_skill_planner import segment_observation
from cloth_agent.grounded_sam2 import close_worker


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observation',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():p.error('output must be a new directory')
    roi=json.loads(Path('config/topdown.unfold_zed.json').read_text(encoding='utf-8'))
    view=reconstruct(Observation.load(a.observation),out,roi)
    originals=[v.copy() for v in (view.xyz,view.source_xy,view.camera_ids)]
    try:
        _,report=segment_observation(view)
        for old,new in zip(originals,(view.xyz,view.source_xy,view.camera_ids)):
            if not np.array_equal(old,new,equal_nan=True):raise AssertionError('segmentation changed measured geometry')
        print(json.dumps(dict(report,measured_arrays_unchanged=True,robot_calls=0),ensure_ascii=False))
    finally:close_worker()

if __name__=='__main__':main()
