"""Replay saved pixel geometry only: no model requests or hardware imports."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloth_agent.contour_normal_grasp import geometry

def main():
    sample = ROOT / 'reproduction/dual_wrist_unfold_20261114/sample'
    selection = json.loads((sample/'selection.json').read_text())
    expected = json.loads((sample/'expected.json').read_text())
    mask = np.asarray(Image.open(sample/'mask.png').convert('L')) > 0
    got = geometry(mask, selection['boundary_point_px'], **expected['options'])
    for name, value in expected['geometry'].items():
        np.testing.assert_allclose(got[name], value, atol=1e-6)
    print(json.dumps(dict(status='PASS', grasp_px=got['grasp_px'], target_px=got['target_px'],
                         robot_calls=0, model_calls=0, scope='saved pixel geometry only')))

if __name__ == '__main__':
    main()
