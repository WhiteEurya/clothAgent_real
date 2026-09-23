"""Project execution telemetry into model context without duplicating raw traces."""
import copy

ACTION_FIELDS = ('name', 'args', 'requested_at', 'completed_at', 'success', 'error', 'actual_ee_pose')


def compact_context(value, key=''):
    if isinstance(value, list):
        if key in {'executed_actions', 'actual_robot_actions'}:
            return [{k: compact_context(v, k) for k, v in action.items() if k in ACTION_FIELDS}
                    if isinstance(action, dict) else action for action in value]
        return [compact_context(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            # Retry history belongs at the top level, not recursively in geometry.
            if k == 'height_retry' and isinstance(v, dict):
                result[k] = compact_context({a: b for a, b in v.items() if a != 'attempt_history'})
            else:
                result[k] = compact_context(v, k)
        return result
    return copy.deepcopy(value)
