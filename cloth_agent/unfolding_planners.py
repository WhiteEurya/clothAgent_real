"""Four planner layers, with a shadow gate and explicitly enabled error recovery."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil

from .experiments import one_shot_goal_planner_probe_v1 as wire


class PlannerState(str, Enum):
    FAST_ACTION = "FAST_ACTION"
    FAST_DONE = "FAST_DONE"
    FAST_ABSTAIN = "FAST_ABSTAIN"
    HOST_REJECT = "HOST_REJECT"
    GATE_TRUST = "GATE_TRUST"
    GATE_DELIBERATE = "GATE_DELIBERATE"
    GATE_UNCERTAIN = "GATE_UNCERTAIN"
    DELIBERATIVE_ACTION = "DELIBERATIVE_ACTION"
    MEMORY_ACTION = "MEMORY_ACTION"


@dataclass(frozen=True)
class DeliberativeContext:
    current_rgb: Path
    goal_rgb: Path
    current_rgbd_directory: Path | None = None
    fast_proposal: dict | None = None
    compact_episode_history: tuple = ()
    host_uncertainty_summary: dict | None = None


@dataclass(frozen=True)
class MemoryContext:
    observation: DeliberativeContext
    frozen_trajectory_manifest: Path | None = None
    graph_reference: Path | None = None


class PlannerDisabledError(RuntimeError):
    pass


class DeliberativePlanner:
    enabled = False

    def plan(self, context: DeliberativeContext) -> dict:
        """Future ACTION replacement or ABSTAIN, using the FastPlanner action schema."""
        raise PlannerDisabledError("DeliberativePlanner is disabled in shadow v1")


class MemoryPlanner:
    enabled = False

    def plan(self, context: MemoryContext) -> dict:
        """Future ACTION replacement or ABSTAIN; no memory is opened in v1."""
        raise PlannerDisabledError("MemoryPlanner is disabled in shadow v1")


class FastPlanner:
    """Adapter around the unchanged one-shot request preparation and transport."""
    def __init__(self, plan_once):
        self._plan_once = plan_once

    def plan(self, root, folder, before, headers, timeout_s):
        valid = False
        try:
            plan, metrics = self._plan_once(root, folder, before, headers, timeout_s)
            valid = True
            return plan, metrics
        finally:
            # Legacy names remain for existing audit/report scripts.
            for old, new in (("plan.json", "fast_plan.json"),
                             ("proposed_action_overlay.png", "fast_action_overlay.png")):
                if (folder/old).exists():
                    shutil.copyfile(folder/old, folder/new)
            wire.save(folder/"fast_status.json", dict(valid=valid, model=wire.MODEL,
                effort=wire.EFFORT, retries=0, automatic_slow_routing=False))


class ShadowPlannerRouter:
    """Gate never routes; only explicitly enabled recovery can replace a fast plan."""
    def __init__(self, folder, *, recovery_enabled=False):
        self.folder = Path(folder)
        self.recovery_enabled = recovery_enabled
        self._replacement = None
        self.events = []
        self._fast = None
        self.record()

    def record(self, state=None, **detail):
        if state is not None:
            self.events.append(dict(state=PlannerState(state).value, at=wire.utc(), **detail))
        wire.save(self.folder/"planner_states.json", dict(
            mode="FAST_WITH_HOST_RECOVERY_AND_SHADOW_GATE" if self.recovery_enabled else "FAST_WITH_SHADOW_GATE",
            execution_source="CodexRecovery" if self._replacement is not None else "FastPlanner",
            automatic_escalation=self.recovery_enabled, escalation_trigger="allowlisted pre-dispatch errors only",
            deliberative_enabled=self.recovery_enabled, memory_enabled=False, events=self.events))

    def accept_fast(self, plan):
        self._replacement = None
        self._fast = deepcopy(plan)
        self.record("FAST_"+plan["decision"])

    def accept_deliberative(self, plan):
        if not self.recovery_enabled:
            raise PlannerDisabledError("Recovery was not explicitly enabled")
        self._replacement = deepcopy(plan)
        self.record("DELIBERATIVE_ACTION" if plan["decision"] == "ACTION" else None)

    @property
    def execution_plan(self):
        if self._replacement is not None:
            return deepcopy(self._replacement)
        if self._fast is None:
            raise RuntimeError("No validated fast plan")
        return deepcopy(self._fast)

    def shadow(self, gate, current_rgb, goal_rgb):
        # A gate implementation cannot mutate the object later sent to Host.
        result = gate.evaluate(current_rgb, goal_rgb, deepcopy(self._fast))
        self.record("GATE_"+result["decision"], shadow=True,
                    result_status=gate.metrics["status"])
        return result

    def host_reject(self, reason):
        # Never create DELIBERATIVE_ACTION or infer a policy-quality label here.
        self.record(PlannerState.HOST_REJECT, reason=reason, policy_failure_inferred=False)
