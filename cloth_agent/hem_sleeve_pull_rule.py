"""Opt-in action-purpose rule; not a change to geometry or robot safety."""
import json
from pathlib import Path

CONFIG='hem_sleeve_pull_rule.json'
RULE='''
【已启用独立实验规则 --hem-sleeve-pull-rule】
如果目的是立即展开下摆，就抓取可见下摆并拉开下摆，Pb服务于下摆展开，不改为拉衣袖来间接替代。若目的是显露单侧隐藏下摆角，则遵循显露衣角的特例，不强制向外拉开下摆。
如果目前的袖子可以被摩擦力拉开,则抓对向
'''


def apply(root,prompt):
    path=Path(root)/CONFIG
    if not path.exists():return prompt
    config=json.loads(path.read_text())
    if config.get('rgb_only') is True:
        prompt='\n'.join(line for line in prompt.splitlines() if not line.startswith(('几何报告仅作辅助','结构距离报告')))
        prompt=prompt.replace('附带Geometry Report来自本轮RGB-D的本地计算，不臆造像素到毫米换算；几何可行性仍由Host检查。','几何可行性仍由Host检查，不臆造像素到毫米换算。')
    if config.get('enabled') is not True:return prompt
    # Remove older opposite-side triggers rather than stacking conflicting rules.
    prompt='\n'.join(line for line in prompt.splitlines() if not line.startswith('若目标袖子或下摆角不可见'))
    prompt=prompt.replace('优先改选清晰可见的对侧衣袖/下摆，无可靠抓点则ABSTAIN。','无可靠抓点则ABSTAIN。')
    prompt=prompt.replace('若因身份不可靠改选对侧，点明原因。','')
    return prompt+'\n'+RULE


def rgb_only(root):
    path=Path(root)/CONFIG
    return path.exists() and json.loads(path.read_text()).get("rgb_only") is True
