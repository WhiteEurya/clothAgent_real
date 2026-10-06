# 测试目录

项目的测试代码和测试样例统一放在此目录：

- `test_*.py`：自动回归测试，从项目根目录运行 `python -m pytest`。
- [manual/](manual/README.md)：手动专项测试，使用 `python tests/manual/<脚本名>.py` 按需运行。
- `manual/legacy/`：原项目根目录的三个旧远程连接测试，保留原脚本和参数；其中 `remote_planner_test.py` 与新版手动测试分开保存。
- `fixtures/`：测试图片 `test.png` 和已记录的夹爪轨迹样例。

`conftest.py` 排除手动测试和样例目录的自动收集，避免普通 pytest 运行触发相机、机器人或远程服务。旧远程脚本按原行为执行，不支持统一的 `--help`。

抖动展开的正式动作实现位于 `cloth_agent/shake_open.py`，手动测试入口为 `tests/manual/xarm_shake_open_test.py`。
