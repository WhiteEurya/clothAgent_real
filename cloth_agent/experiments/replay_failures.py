"""Classify existing executor errors for reporting, without relaxing any check."""
import re
import json
from pathlib import Path
import traceback

_CONTROLLER = re.compile(r'\b(left|right): controller fault/state (\d+),\s*\[(\d+),\s*(\d+)\]')


def controller_fault_details(error):
    """Recognize the exact snapshot error emitted by dual_arm.execution."""
    match=_CONTROLLER.search(str(error))
    if not match:return None
    arm,state,code,warning=match.groups()
    return dict(arm=arm,controller_state=int(state),error_code=int(code),warning_code=int(warning),
                recovery='STOP_NO_AUTOMATIC_CLEAR_OR_REDISPATCH',raw_error=str(error))


def record_controller_failure(result,error):
    detail=controller_fault_details(error)
    if detail is None:return False
    result.update(status='CONTROLLER_FAULT',error=str(error),controller_fault=detail)
    return True


def stop_after_failure(runtime,result):
    try:runtime.stop()
    except Exception as error:
        result['stop_error']=f'{type(error).__name__}: {error}'


def capture_cli_failure(output,error,*,record_name='final_metrics.json'):
    """Keep a replay's evidence and expose failures before/after its own try block."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    try:
        result=json.loads((output/record_name).read_text())
        if not isinstance(result,dict):result={}
    except (OSError,ValueError):result={}
    detail=f'{type(error).__name__}: {error}'
    if not result.get('error'):
        result.update(status='INTERRUPTED' if isinstance(error,KeyboardInterrupt) else 'ERROR',error=detail)
    elif result['error'] != detail:
        result['propagated_error']=detail
    record_controller_failure(result,result['error'])
    trace=output/'cli_exception.txt'
    trace.write_text(''.join(traceback.format_exception(type(error),error,error.__traceback__)))
    result['exception_trace']=str(trace.resolve())
    return result


def close_cli_runtime(runtime,result):
    """Always attempt preview shutdown; cleanup must not hide the primary error."""
    if runtime is None:return
    for name,resource in [('runtime',runtime),('preview',getattr(runtime,'preview',None))]:
        if resource is None:continue
        try:resource.close()
        except Exception as error:
            result.setdefault('cleanup_errors',[]).append(dict(resource=name,error=f'{type(error).__name__}: {error}'))


def cli_failed(result):
    return (result.get('status') in {'ERROR','INTERRUPTED','CONTROLLER_FAULT','SAFETY_REJECT','ACTION_EXECUTION_FAILURE',
                                   'FALLBACK_FAILED','APPLICABILITY_FRONTIER_FAILED','RECOVERY_UNRESOLVED'}
            or bool(result.get('exception_trace') or result.get('cleanup_error') or result.get('cleanup_errors')))
