"""Pinned RAX merge compatibility: advertise the sanitized scratch commit.

Upstream 36c926c copies objects into agent.git but leaves all refs empty.
Git 2.25 refuses fetching a SHA from this empty-advertisement repository.
Keep upstream merge/validation unchanged; add only a ref to its scratch copy.
"""
from pathlib import Path
import re


def install():
    from services import skill_library
    original = skill_library._git
    if getattr(original, '_rax_scratch_ref_compat', False):
        return

    def git(*args, **kwargs):
        if (len(args) == 5 and args[:3] == ('fetch', '-q', '--no-tags')
                and re.fullmatch(r'[0-9a-f]{40}', args[4])):
            scratch = Path(args[3])
            # Only the harness-created sanitized repository, never the agent clone.
            if (scratch.name == 'agent.git' and scratch.parent.name.startswith('skill-merge-')
                    and not scratch.is_symlink()
                    and (scratch/'config').read_text() == '[core]\n\trepositoryformatversion = 0\n\tbare = true\n'
                    and (scratch/'HEAD').read_text() == 'ref: refs/heads/main\n'):
                original('--git-dir', str(scratch), 'update-ref', 'refs/heads/main', args[4])
        return original(*args, **kwargs)

    git._rax_scratch_ref_compat = True
    skill_library._git = git
