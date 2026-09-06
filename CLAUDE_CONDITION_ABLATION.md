# Claude condition ablation on saved rollouts

`scripts/claude_condition_ablation.py` is an offline experiment for measuring
how much each evidence source changes Claude's garment-state judgment. It does
not access RealSense, the xArm, or the robot execution loop.

For every discovered `recording_manifest.json`, it extracts stills from the
saved A/B RGB videos, marks the historical selected candidate when the segment
has one, and asks the same ten fixed questions under four conditions:

| Condition | Evidence sent to Claude |
| --- | --- |
| A | Camera-A before/after RGB only |
| B | A plus Camera-B before/after RGB |
| C | B plus saved height/depth visualizations |
| D | A/B before/during/after RGB, height/depth visualizations, and historical action metadata |

The weaker conditions are intentionally static. For acquisition and motion
questions, Claude should return `UNKNOWN` when before/after evidence cannot
prove the intermediate event. Condition D is the only condition that receives
action metadata, and the prompt explicitly says that commanded actions are not
evidence of acquisition.

## Run

Start with an evidence-only audit:

```bash
python scripts/claude_condition_ablation.py \
  --runs-root runs/claude_global_cli_real_20260821_233747 \
  --limit 2 \
  --dry-run \
  --output-dir /tmp/claude_condition_ablation_smoke
```

Then run the four Claude calls per segment (50 segments means 200 calls):

```bash
python scripts/claude_condition_ablation.py \
  --runs-root runs \
  --limit 50 \
  --seed 20260828 \
  --output-dir runs/claude_condition_ablation_50
```

Useful options:

- `--recording PATH` selects an explicit recording directory or manifest.
- `--conditions A B` runs only selected conditions while developing prompts.
- `--resume` reuses completed response files after an interrupted run.
- `--limit 0` processes every discovered segment.
- `--fail-fast` stops on the first bundle or Claude error.

Outputs include `experiment_index.json`, extracted evidence bundles, one prompt
and response per segment/condition, `results.jsonl`, and `summary.json`. The
summary reports YES/NO/UNKNOWN counts and mean confidence for each question and
condition, plus per-question label transitions for every condition pair (for
example, how often D changes an A `UNKNOWN` into a known answer).
