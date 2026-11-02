"""Binary permission result; NO does not assert a proven physical collision."""
import json
from pathlib import Path

def execution_permission(result):
    return 'YES' if (
        result.get('status') == 'SAFETY_PLAN_PASSED_PENDING_HUMAN_APPROVAL'
        and result.get('path_validation') is True
        and result.get('disabled_collision_envelopes') == []
        and isinstance(result.get('operation_count'), int)
        and result['operation_count'] > 0
        and not result.get('error')
    ) else 'NO'

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args()
    try:
        result=json.loads(a.result.read_text())
        answer=execution_permission(result) if isinstance(result,dict) else 'NO'
    except (OSError,ValueError,TypeError):
        answer='NO'
    print(answer)

if __name__=='__main__':main()
