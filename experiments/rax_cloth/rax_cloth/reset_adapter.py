"""Host-only reuse of the existing environment reset, not an Agent skill."""
from pathlib import Path
import time


def run_existing_reset(port, folder, seed, authorize, dispatch):
    from cloth_agent.codex_native_runtime import NativeRuntime
    from cloth_agent.single_arm_runtime import SingleArmRuntime
    from cloth_agent.random_fold_reset import run_reset

    class ResetRuntime(NativeRuntime):
        # Reuse only the existing single-arm preparation recipe; never its
        # collection runtime's safety-disabling configuration overrides.
        arm_key='left'
        robot_name='arm6'
        camera_label='A'
        _preparation_program=SingleArmRuntime._preparation_program

        def plan(self, source, initial, **kwargs):
            authorize()
            from cloth_agent.codex_motion import parse_code
            for name,args in parse_code(source):
                if 'right' in args or args.get('arm')=='right':
                    raise PermissionError('reset may only command arm6')
            began=time.monotonic()
            operations=super().plan(source,initial,**kwargs)
            certified[id(operations)]=(operations,initial,time.monotonic()-began)
            return operations

        def _execute_operations(self,operations,directory):
            authorize()
            record=certified.pop(id(operations),None)
            if record is None or record[0] is not operations:
                raise PermissionError('reset operations were not certified by the existing planner')
            self.arrival(record[1])
            return dispatch({'status':'SAFE','operations':operations},record[2],Path(directory))

        def capture_current(self,directory):
            authorize()
            return port.capture('camA',directory)

    certified={}
    runtime=ResetRuntime.__new__(ResetRuntime)
    # Share the existing controller, cancellation and preparation state; no
    # second controller owner or SDK connection is created by this adapter.
    runtime.__dict__=port.runtime.__dict__
    authorize()
    # Keep existing reset parameters/defaults, retries, VLM prompts, recorder,
    # seed sampling and before/after evaluation exactly in run_reset.
    return run_reset(runtime,folder,seed=seed)
