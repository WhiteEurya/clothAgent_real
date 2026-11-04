"""Dependency-free State → Action → Outcome export when Graphviz is absent."""
from html import escape


def render_growth_svg(state, path):
    rows = []
    for aid, action in state['actions'].items():
        for target, outcome in action['outcomes'].items():
            rows.append((action['source_node'], aid, target, outcome['observed_count']))
        if not action['outcomes']:
            rows.append((action['source_node'], aid, '(unobserved)', 0))
    covered = {r[0] for r in rows} | {r[2] for r in rows}
    rows.extend((node, '', '', '') for node in state['nodes'] if node not in covered)
    height = max(120, 80 + 60 * len(rows))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="25" font-family="sans-serif" font-size="16">State → Action → Outcome (one row per outcome; repeated IDs are the same node)</text>']
    for i, (source, action, target, count) in enumerate(rows):
        y = 50 + i * 60
        labels = (source, action, f'{target}  n={count}' if target else '')
        if action:
            parts.append(f'<path d="M300 {y+15}H370 M650 {y+15}H720" stroke="#666" fill="none"/>')
        for x, label in zip((20, 370, 720), labels):
            if not label:
                continue
            parts.append(f'<rect x="{x}" y="{y}" width="280" height="32" rx="5" fill="#e4eef7" stroke="#8899aa"/>')
            parts.append(f'<text x="{x+8}" y="{y+21}" font-family="sans-serif" font-size="12">{escape(str(label))}</text>')
    path.write_text('\n'.join(parts + ['</svg>']) + '\n')
