#!/usr/bin/env python3
"""Generate an offline smooth candidate from saved native_plan.json; no robot API."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.continuous_motion import compile_continuous


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compile_continuous(json.loads(args.plan.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    groups=[g for g in result['groups'] if g['kind']=='continuous_candidate']
    print(json.dumps(dict(output=str(args.output),execution_authorized=False,
        groups=len(groups),internal_knots=sum(g['internal_knots'] for g in groups),
        moving_internal_knots=sum(g['moving_internal_knots'] for g in groups),
        candidate_motion_duration_s=sum(g['duration_s'] for g in groups)),indent=2))

if __name__=='__main__':main()
