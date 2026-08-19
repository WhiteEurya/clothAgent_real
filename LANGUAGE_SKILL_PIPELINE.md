# Language-to-skill garment pipeline

This experiment is implemented only in `cloth_agent/language_skill_pipeline.py`.
It does not modify the existing free- or automatic-exploration pipelines.

The pipeline performs one bounded experiment:

1. Ask Claude to author a task-specific `generated_skills/<name>/SKILL.md` from the language instruction.
2. Capture calibrated Camera A/B RGB-D evidence, or reuse evidence from an existing run.
3. Optionally run the existing MolmoPoint worker with the visual query selected by that skill.
4. Start a fresh, read-only Claude process that uses the generated skill, images, Molmo output, and one calibrated grounding lookup to emit restricted RobotAPI actions.
5. Run the existing static preflight and execute in simulation by default. A real rollout still requires explicit flags and the normal controller/TCP/IK gates.

Create a run and simulate the generated action:

```bash
python -m cloth_agent.language_skill_pipeline \
  --run-id sleeve_outward_01 \
  --instruction "抓住袖子往外移动"
```

Enable the skill-selected Molmo query:

```bash
python -m cloth_agent.language_skill_pipeline \
  --run-id sleeve_outward_molmo_01 \
  --instruction "抓住袖子往外移动" \
  --use-molmo \
  --molmo-python /home/CNS2026330003/miniconda3/envs/molmo/bin/python
```

Reuse an existing run's `workspace/perception_views` without opening the cameras:

```bash
python -m cloth_agent.language_skill_pipeline \
  --run-dir runs/existing_run \
  --skip-capture \
  --instruction "抓住袖子往外移动"
```

Simulation is the default. For a physical rollout, review the printed plan and
run with both `--real` and `--confirm-real`. The result directory stores the
instruction, generated skill, optional Molmo output, both Claude invocation
records, validated plan, restricted experiment source, and execution result.
