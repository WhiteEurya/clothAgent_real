"""Thin local operator CLI using upstream services.arming, never a robot service."""
import argparse
import json
import os
from pathlib import Path
import sys
from .baseline import register


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('status','arm','disarm','requests','reply'))
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--minutes',type=int,default=1)
    p.add_argument('--request-id')
    p.add_argument('--text',default='done')
    a=p.parse_args()
    register()
    os.environ['RAX_ARTIFACT_ROOT']=str(a.session.resolve().parent)
    from services import arming
    from services.controller.config import Configuration
    config=Configuration.load(a.session/'configuration.json')
    if config.backend_options.get('mode')!='real_supervised':p.error('not a real_supervised session')
    if a.action in ('requests','reply'):
        from services.controller import operator
        if a.action=='requests':result=operator.pending(config.root)
        else:
            if not a.request_id:p.error('--request-id required for reply')
            result=operator.reply(config.root,a.request_id,a.text,by='xarm-operator-cli')
    elif a.action=='arm':
        if not 1<=a.minutes<=30:p.error('--minutes must be 1..30')
        if not sys.stdin.isatty():p.error('onsite operator terminal required')
        if input('现场操作员确认急停可用并在场，输入 ARM 授权：').strip()!='ARM':raise SystemExit('cancelled')
        result=arming.write(config.root,True,minutes=a.minutes)
    elif a.action=='disarm':result=arming.write(config.root,False,reason='operator disarm')
    else:result=arming.read(config.root)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
