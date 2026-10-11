"""Apply the user's shared arm7 grasp-height policy before native planning."""
import ast
import copy

from .codex_motion import parse_code


def share_arm7_grasp_z(source):
    calls=parse_code(source)
    if calls[0][0]=='blocked':return source,[]
    closes=[i for i,(name,args) in enumerate(calls) if name=='gripper' and args['action']=='close']
    if len(closes)%2:
        raise ValueError('shared arm7 grasp Z requires a left/right close pair')
    tree=ast.parse(source);changes=[]
    # Complete all validation before producing or planning modified code.
    for first,second in zip(closes[::2],closes[1::2]):
        pair={calls[i][1]['arm']:i for i in (first,second)}
        if set(pair)!={'left','right'}:
            raise ValueError('shared arm7 grasp Z requires one close per arm')
        first_arm=calls[first][1]['arm']
        for name,args in calls[first+1:second]:
            if ((name.startswith('move_') and first_arm in args)
                    or (name=='gripper' and args['arm']==first_arm)):
                raise ValueError('both claws must close before moving/releasing the first grasp')
        targets={}
        for arm,index in pair.items():
            for j in range(index-1,-1,-1):
                name,args=calls[j]
                if name=='gripper' and args['arm']==arm and args['action']=='close':
                    raise ValueError('new grasp requires an explicit TCP target')
                if name.startswith('move_') and arm in args:
                    if name!='move_tcp':
                        raise ValueError('shared grasp height requires move_tcp before close, not joint targets')
                    targets[arm]=(j,copy.deepcopy(args[arm]));break
            else:raise ValueError(f'{arm}: no explicit grasp TCP target before close')
        index,left=targets['left'];_,right=targets['right']
        old=left[2];left[2]=right[2]
        node=tree.body[index].value
        keyword=next(k for k in node.keywords if k.arg=='left')
        keyword.value=ast.parse(repr(left),mode='eval').body
        changes.append(dict(left_move_index=index,left_close_index=pair['left'],right_close_index=pair['right'],
                            original_left_z_mm=old*1000,arm7_reference_z_mm=right[2]*1000,
                            applied_left_z_mm=left[2]*1000))
    if not changes:return source,[]
    return ast.unparse(ast.fix_missing_locations(tree))+'\n',changes


def lower_grasp_z(source,*,arm,offset_mm):
    """Lower the last explicit TCP target before each new close, in World Z.

    Reset at gripper events/joint moves so a repeated close cannot lower an
    earlier target again. Approach, transport, release and lift targets stay put.
    """
    import math
    if arm not in ('left','right') or not math.isfinite(offset_mm) or offset_mm<0:
        raise ValueError('invalid grasp lowering policy')
    calls=parse_code(source);tree=ast.parse(source);pending=None;changes=[]
    for index,(name,args) in enumerate(calls):
        if name in ('move_tcp','move_joints') and arm in args:
            pending=index if name=='move_tcp' else None
        elif name=='gripper' and args['arm']==arm:
            if args['action']=='close' and pending is not None:
                target=list(calls[pending][1][arm]);old=target[2];target[2]-=offset_mm/1000
                keyword=next(k for k in tree.body[pending].value.keywords if k.arg==arm)
                keyword.value=ast.parse(repr(target),mode='eval').body
                changes.append(dict(arm=arm,move_index=pending,close_index=index,
                    original_z_mm=old*1000,applied_z_mm=target[2]*1000))
            pending=None
    return (ast.unparse(ast.fix_missing_locations(tree))+'\n' if changes else source),changes
