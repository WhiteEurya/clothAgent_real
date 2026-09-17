# 关节 1、2 S0：本机日志调查（2026-09-15）

## 结论

用户提供的 Studio 截图显示 S0 Joint Communication Error，Joint ID [1,2]；用户确认退出程序后仍发生，且每次编号相同。本机可访问记录中未找到对应的 S0 事件，不能据此确定通信线、供电、驱动器或固件中的具体故障点。

## 检查范围与证据

- 解析 runs/、results/ 下 1,250 个可解析且不超过 5 MB 的 JSON 文件。发现 166 处 error_warn 快照，均为 [0,[0,0]]；包含重复保存，不能视为 166 个独立事件，也不覆盖程序退出后的时段。
- 使用包含 gitignore 文件的搜索检查 runs/、results/、artifacts/ 的日志、文本、JSON/JSONL；未找到 ServoError、ControllerError、Joint Communication Error 等对应故障文字。
- 用户目录的 Downloads、Documents、Desktop、.config、.local/share，以及 /opt、/tmp 的文件名检索中，未识别出 UFACTORY Studio 故障日志或导出的机器人诊断包；未发现 ~/.ros 日志目录。
- /var/crash 为空；可访问的 9 月 12 日以来用户 journal 中未匹配到 xArm/UFActory/servo、进程段错误、OOM 或网络链路掉线关键词。用户 journal 不等于完整系统日志。
- /var/log/syslog、syslog.1、kern.log、kern.log.1 均因 Linux 权限无法读取。提升沙箱权限后仍不可读，sudo -n 提示需要密码；未完成系统/内核日志排查。
- 本机 robo 环境 xArm SDK 1.18.4 的 xarm/core/utils/log.py 使用 StreamHandler(sys.stdout)，未配置独立文件日志。因此不能期待默认存在 SDK 历史故障文件。

## 发现的其他异常

- 2026-09-14 的 fold_20260914T114526780131945Z：set_gripper_mode failed, code=22，发生在感知前初始化，两次尝试均失败。这是夹爪接口返回码，不是截图中的关节 S0。
- run_20260914T113059Z、fold_20260914T114737548435928Z、fold_20260915T014647067470868Z：TCP offset 的 0 与 172 mm 不匹配；其中既有期望 172、读取 0，也有相反方向。日志不能区分配置变更、读取未就绪或控制器参数变化，不能据此认定掉电。
- 2026-09-15 13:23:56 +08：夹爪打开完成检查失败，位置 67，目标 850，夹爪错误码 0；随后 mandatory_return_home 完成。不能作为关节 1、2 通信故障的证据。
- 最新检查的 fold_20260915T073447547452550Z/debug.log 结束原因为规划阶段 KeyboardInterrupt，未记录伺服故障。不能据此推断中断的外部原因。

## 缺失证据

仍需要控制箱/Studio 导出的故障历史（关节通信与伺服诊断），以及故障准确时间。已有项目快照主要发生于程序运行期间，无法还原退出程序后的 S0。没有控制器侧记录时，不能区分关节通信中断、驱动器复位和供电异常。

本次仅检索本机文件；未连接或操控机械臂，未清错、使能或更改控制参数。
