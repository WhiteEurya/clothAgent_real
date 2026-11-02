"""Deterministic factual feedback; never recommends a next grasp."""
def compress(outcome,action):
    c,m,e=outcome.coverage,outcome.motion,outcome.execution
    pieces=[f'Previous executed Pa={action.pick_px}, Pb={action.place_px}.',f'Execution={e["status"]}; cloth grasp=UNKNOWN.']
    if c['delta_ratio'] is None:pieces.append('Projected area change=UNKNOWN.')
    else:pieces.append(f'Projected area change={100*c["delta_ratio"]:+.1f}% ({c["status"]}; not unfolding progress).')
    if m['status']=='OK':pieces.append(f'Rigid-fit contour IoU={m["aligned_iou"]:.2f}; center motion={m["translation_px"]:.1f}px; rotation={m["rotation_deg"]:.1f}deg (provisional).')
    else:pieces.append('Rigid motion=UNKNOWN.')
    pieces.append('Local unfolding=UNKNOWN.')
    return ' '.join(pieces)
