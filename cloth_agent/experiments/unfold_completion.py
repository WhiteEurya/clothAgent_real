"""Shared completion wording for skill and plain unfolding planners."""
PREVIOUS_DONE_RULE = '衣物整体已基本展开、没有明显大面积折叠或覆盖时，返回DONE。仅当下一步预计能明显改善整体展开程度时才继续；不要为局部皱褶、袖口形状、收窄或轻微不对称反复拉扯，避免破坏已经展开的整体形态。'
STRETCH_LENGTH_DONE_RULE = '衣身和两侧袖子均已展开，袖子不再明显收在衣身内，且没有明显大面积折叠或覆盖时，返回DONE。袖子尚未展开或仍被压在衣身内时不能视为完成。衣领必须显露：衣领区域可辨认、未被大块布层覆盖即可；不要求领口撑开，也不要求完整领圈清晰可见。允许局部皱褶和轻微不对称，但袖子相对衣身的伸出长度应尽量接近目标图；若袖子仍明显短缩，应继续展开，不能仅因袖口可见就判定DONE。伸出程度已接近目标后，不要仅为消除褶皱继续拉扯。'
# Adopted from the Arm6 offline comparison (2026-11-12), condition B.
ARM6_DONE_RULE = '只做一个简单完成判断：主要衣身、双袖和下摆已基本展开、没有明显内收或大片覆盖时DONE，允许普通小皱纹；仍需展开时ACTION；无法可靠选择动作时ABSTAIN。'
DONE_RULE = ARM6_DONE_RULE
OLD_RULES = tuple('只有当前图中衣物四个外轮廓角都清晰可见、没有角被遮住或卷入'+sep+
                 '没有明显大片覆盖或内收时才允许DONE；只要任一角不可见、被遮挡、卷入或无法确认，就继续选择动作。'
                 for sep in ('，且','、且'))

def update_completion(prompt):
    prompt = prompt.replace(PREVIOUS_DONE_RULE, DONE_RULE)
    prompt = prompt.replace(STRETCH_LENGTH_DONE_RULE, DONE_RULE)
    for old in OLD_RULES:
        prompt = prompt.replace(old, DONE_RULE)
    return prompt.replace('局部普通皱褶不妨碍动作，但不能据此提前DONE。', '')
