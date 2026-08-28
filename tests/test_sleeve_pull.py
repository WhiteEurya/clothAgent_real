from __future__ import annotations

import scripts.claude_sleeve_pull as sleeve_pull


def test_sleeve_pull_wrapper_injects_the_sleeve_task(monkeypatch) -> None:
    seen = {}

    def fake_main(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(sleeve_pull, "language_skill_main", fake_main)

    assert sleeve_pull.main(["--run-id", "sleeve_pull_test"]) == 0
    args = seen["args"]
    assert args[0] == "--instruction"
    assert "袖子" in args[1]
    assert "最远" in args[1]
    assert args[-2:] == ["--claude-timeout-s", "900"]


def test_sleeve_pull_wrapper_preserves_an_explicit_instruction(monkeypatch) -> None:
    seen = {}

    def fake_main(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(sleeve_pull, "language_skill_main", fake_main)

    assert sleeve_pull.main(["--instruction", "只拉右袖", "--run-id", "sleeve_pull_test"]) == 0
    assert seen["args"] == [
        "--instruction",
        "只拉右袖",
        "--run-id",
        "sleeve_pull_test",
        "--claude-timeout-s",
        "900",
    ]
