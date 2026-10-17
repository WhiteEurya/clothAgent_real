"""Host progress labels derived solely from reliable decision-state matches."""

def node_match_progress(source, expected_target, result, *, reliable=True, conflict=False):
    if conflict or not reliable or source is None or result is None:
        outcome='UNKNOWN'
    elif result==source:
        outcome='NO_STATE_CHANGE'
    elif expected_target is not None and result==expected_target:
        outcome='PROGRESS'
    else:
        outcome='ALTERNATIVE_OUTCOME'
    return dict(progress=outcome,basis='NODE_MATCH_ONLY',source_node=source,
                expected_target=expected_target,result_node=result,match_conflict=conflict,
                limitation='No state change does not establish an empty grasp or absence of physical motion.')
