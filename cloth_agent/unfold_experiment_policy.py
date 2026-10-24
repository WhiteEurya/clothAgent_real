"""Accounting for consecutive observed outcomes and radius-sweep trials."""

def next_streak(streak, verdict, uncertain_is_no_improvement=False):
    if verdict in ('UNCHANGED','WORSE') or (verdict=='UNCERTAIN' and uncertain_is_no_improvement):
        return streak+1
    if verdict in (None,'IMPROVED','UNCERTAIN'):
        return 0
    raise ValueError('unknown previous-effect verdict: '+str(verdict))


def stop_reason(decision, streak, patience, completed_actions, max_actions):
    # Judge DONE on the outcome of the last action before declaring stagnation.
    if decision=='DONE':return None
    if patience and streak>=patience:return 'NO_IMPROVEMENT'
    if max_actions and completed_actions>=max_actions:return 'ROUND_LIMIT'
    return None


def classify(record, returncode=0, operator_stopped=False):
    if operator_stopped or returncode!=0 or not record.get('finished_at') or record.get('cleanup_error'):
        return 'INVALID'
    reason=record.get('termination_reason')
    if reason=='DONE':return 'SUCCESS'
    if reason in ('NO_IMPROVEMENT','ROUND_LIMIT'):return 'FAILURE'
    return 'INVALID'


def radius_summary(records, radius):
    attempts=[r for r in records if r['d_mm']==radius]
    valid=[r for r in attempts if r['outcome']!='INVALID']
    successes=sum(r['outcome']=='SUCCESS' for r in valid)
    return dict(d_mm=radius,attempts=len(attempts),valid_trials=len(valid),successes=successes,
                failures=len(valid)-successes,invalid_trials=len(attempts)-len(valid),
                success_rate=successes/len(valid) if valid else None,
                complete=len(valid)==3,zero_success_stop=len(valid)==3 and successes==0)
