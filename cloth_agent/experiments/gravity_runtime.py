"""Experiment-only arm7 adapter; reuse all native planning and safety checks."""
from dataclasses import replace
from pathlib import Path
from copy import copy, deepcopy

from cloth_agent.codex_motion import parse_code

from cloth_agent.codex_arm7_runtime import Arm7Runtime


class HangingObservation:
    """Diagnostic RGB-D with explicit unpatched calibration provenance."""
    def __init__(self, observation, capture_config, parent_calibration):
        self.observation = observation
        self.capture_config = capture_config
        self.parent_calibration = parent_calibration
        self.directory = observation.directory
        self.image = observation.image
        self.meta = observation.meta

    def validate_for(self, config, **kwargs):
        if config.raw['calibration_id'] != self.parent_calibration:
            raise ValueError('hanging observation parent calibration changed')
        self.observation.validate_for(self.capture_config, **kwargs)

    def sample(self, *args, **kwargs):
        raise ValueError('diagnostic hanging depth cannot supply a new grasp')


class GravityRuntime(Arm7Runtime):
    grasp_offset_mm = 0.
    holding = False
    primitive = None

    def load_config(self, config_path):
        config = super().load_config(config_path)
        # Experiment-local dynamics, applied BEFORE planner/coordinator creation.
        # Speed validation is user-reported; the stopping margin below is an
        # engineering extrapolation, not a newly measured stop excursion.
        config.raw = deepcopy(config.raw)
        config.limits = deepcopy(config.limits)
        baseline = config.limits['joint_speed_deg_s']
        right = config.raw['safety']['arms']['right']
        old_ceiling = right['max_joint_speed_deg_s']
        old_stop = list(right['stop_excursion_deg'])
        speed = baseline * 1.30
        ratio = max(1., speed / old_ceiling)
        # Keep the shared two-arm command floor at its original value. Only
        # right-arm segments are retimed below, using its own validated ceiling.
        right['max_joint_speed_deg_s'] = max(old_ceiling, speed)
        right['stop_excursion_deg'] = [x * ratio ** 2 for x in old_stop]
        config.raw['gravity_speed_policy'] = dict(
            scale=1.30, baseline_joint_speed_deg_s=baseline, joint_speed_deg_s=speed,
            acceleration_changed=False, validation_source='user_reported_130_percent_verified',
            statement='改成130%吧,这个数字我验证过',
            stopping_margin_source='speed_squared_extrapolation_not_new_measurement',
            previous_stop_excursion_deg=old_stop, stop_excursion_deg=right['stop_excursion_deg'])
        return config

    def _join_moves(self, operations, boundaries=()):
        from cloth_agent.dual_arm.sequential_home import command_limits
        operations = super()._join_moves(operations, boundaries)
        command_config = copy(self.config)
        command_config.limits = dict(self.config.limits)
        command_config.limits['joint_speed_deg_s'] = self.config.raw['gravity_speed_policy']['joint_speed_deg_s']
        result = []
        for kind, data in operations:
            if kind == 'move' and data.arm == 'right':
                speed, acceleration, timeout = command_limits(
                    command_config, self.models['right'], self.config.arms['right'],
                    data.start['right'], data.end['right'])
                data = replace(data, speed=speed, acceleration=acceleration, timeout=timeout)
            result.append((kind, data))
        return result

    def _tcp_segments(self, k, pose, q, *, allow_detour=False):
        return super()._tcp_segments(k, pose, q, allow_detour=allow_detour)

    def plan(self, source, initial, **kwargs):
        operations = super().plan(source, initial, preserve_command_boundaries=True)
        if self.primitive == 'LiftAndReveal':
            # The ordinary full pick/place/retreat was validated above. Retain the
            # validated prefix ending BEFORE the host's final opening event.
            closes = [i for i, (kind, data) in enumerate(operations)
                      if kind == 'gripper' and data['action'] == 'close']
            if len(closes) != 1:
                raise ValueError('LiftAndReveal requires one grasp')
            release = next(i for i in range(closes[0] + 1, len(operations))
                           if operations[i][0] == 'gripper' and operations[i][1]['action'] == 'open')
            operations = operations[:release]
        # Scale AFTER native joining, within the existing certified limits.
        # Keep fallback tabletop dragging slow; other primitives use native limits.
        scale = .35 / 1.30 if self.primitive == 'ShortDragReveal' else 1.0
        accel_scale = .35 if self.primitive == 'ShortDragReveal' else 1.0
        return [(kind, replace(data, speed=data.speed * scale,
                               acceleration=data.acceleration * accel_scale, timeout=data.timeout / scale))
                if kind == 'move' else (kind, data) for kind, data in operations]

    def execute_primitive(self, primitive, source, observation, folder):
        if isinstance(observation, HangingObservation):
            calls = parse_code(source, allow_grasp=True)
            if (not self.holding or primitive != 'LiftDrapeRelease'
                    or any(name not in ('move_tcp', 'gripper') for name, _ in calls)
                    or any(name == 'gripper' and args != dict(arm='right', action='open')
                           for name, args in calls)):
                raise ValueError('hanging observation only supports held drape continuation')
        import json
        (Path(folder) / 'speed_policy.json').write_text(
            json.dumps(self.config.raw['gravity_speed_policy'], ensure_ascii=False, indent=2) + '\n')
        self.primitive = primitive
        outcome = super().execute(source, observation, folder)
        if outcome['status'] == 'EXECUTED':
            self.holding = primitive == 'LiftAndReveal'
            if self.holding:
                try:
                    if self.cancel.wait(1.5):
                        raise KeyboardInterrupt('cancelled during hanging dwell')
                except KeyboardInterrupt:
                    self.stop()
                    outcome['status'] = 'INTERRUPTED'
                    return outcome
                outcome['hanging_dwell_s'] = 1.5
        return outcome

    def observe(self, directory):
        if not self.holding:
            return super().observe(directory)
        # Camera B remains at its current calibrated wrist pose. Never call the
        # usual perception preparation here: it opens the jaw and moves Home.
        return self.observe_stationary(directory)

    def observe_stationary(self, directory):
        from cloth_agent.dual_arm.observation import capture
        # The empirical support-plane patch is only valid at perception pose.
        # Capture physical hand-eye geometry here with a distinct identity; do
        # not weaken that patch's pose guard or apply it at the hanging pose.
        config = copy(self.config)
        config.raw = deepcopy(self.config.raw)
        config.raw['camera'].pop('support_height_patch', None)
        parent_calibration = self.config.raw['calibration_id']
        config.raw['calibration_id'] = parent_calibration + '_hanging_unpatched'
        observation = capture(Path(directory), config, self.models,
                              self.coordinator.snapshots, camera=self.camera)
        return HangingObservation(observation, config, parent_calibration)
