# Online experience graph

State → Action → Outcome State. Orange states remain provisional.
Green: preferred nominal; orange: alternative; red: no progress; purple: re-entry; gray: excluded seed evidence. Counts exclude seed priors.

```mermaid
flowchart LR
s0["P000000 [PROVISIONAL]"]
s1["P000001 [PROVISIONAL]"]
s2["P000002 [PROVISIONAL]"]
s3["P000003 [PROVISIONAL]"]
s4["P000004 [PROVISIONAL]"]
s5["P000005 [PROVISIONAL]"]
s6["P000006 [PROVISIONAL]"]
s0 --> a0["F0000000 [preferred]"]
a0 -->|"nominal; NEW_NODE n=1"| s1
s1 --> a1["F0000001 [audit]"]
s0 --> a2["F0000002 [audit]"]
a2 -->|"GLOBAL_REENTRY n=1"| s1
s1 --> a3["F0000003 [audit]"]
a3 -->|"NEW_NODE n=1"| s2
s2 --> a4["F0000004 [audit]"]
a4 -->|"NEW_NODE n=1"| s3
s3 --> a5["F0000005 [audit]"]
a5 -->|"NO_PROGRESS n=1"| s3
s3 --> a6["F0000006 [audit]"]
a6 -->|"NEW_NODE n=1"| s4
s4 --> a7["F0000007 [audit]"]
a7 -->|"NO_PROGRESS n=1"| s4
s4 --> a8["F0000008 [audit]"]
a8 -->|"NO_PROGRESS n=1"| s4
s4 --> a9["F0000009 [audit]"]
a9 -->|"NO_PROGRESS n=1"| s4
s4 --> a10["F0000010 [audit]"]
a10 -->|"NO_PROGRESS n=1"| s4
s4 --> a11["F0000011 [audit]"]
s5 --> a12["F0000012 [preferred]"]
a12 -->|"nominal; NEW_NODE n=1"| s6
s6 --> a13["F0000013 [audit]"]
```

```json
{
  "total_physical_actions": 13,
  "graph_policy_actions": 0,
  "gpt_frontier_actions": 13,
  "recovery_actions": 0,
  "local_outcome_matches": 5,
  "global_reentries": 1,
  "new_provisional_nodes": 7,
  "no_progress_self_loops": 5,
  "multi_outcome_actions": 0,
  "additional_observed_outcomes": 0,
  "current_nodes": 7,
  "current_transitions": 11,
  "graph_action_ratio": 0.0,
  "new_node_rate": 0.38461538461538464,
  "new_node_rate_definition": "post-action new nodes / all physical attempts; initial node discovery excluded",
  "observed_outcome_counts": {
    "F0000000": {
      "P000001": 1
    },
    "F0000001": {},
    "F0000002": {
      "P000001": 1
    },
    "F0000003": {
      "P000002": 1
    },
    "F0000004": {
      "P000003": 1
    },
    "F0000005": {
      "P000003": 1
    },
    "F0000006": {
      "P000004": 1
    },
    "F0000007": {
      "P000004": 1
    },
    "F0000008": {
      "P000004": 1
    },
    "F0000009": {
      "P000004": 1
    },
    "F0000010": {
      "P000004": 1
    },
    "F0000011": {},
    "F0000012": {
      "P000006": 1
    },
    "F0000013": {}
  },
  "unmatched_physical_attempts": 2
}
```
