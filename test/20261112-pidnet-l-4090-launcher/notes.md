# 启动器开发短测记录

- 执行者：terra-worker（gpt-6.1-terra / xhigh）；日期：2026-11-12。
- 命令：`/home/robot/anaconda3/envs/segformer/bin/python -B test/20261112-pidnet-l-4090-launcher/scripts/check_launcher.py`。
- 结果：16 项通过；日志 `logs/check-launcher.log`。测试使用临时模拟数据、模拟设备、短存活子进程，临时文件放在本目录 scripts/ 并清理；没有运行训练或评估。
- 覆盖实际旧配置的两数据集及全部 14 预设、旧配置保存读取、只读预览与缓存、设备遮罩导入顺序、无 CUDA 和模拟显卡拒绝、NumPy 兼容、缺失与错误权重和数据、路径安全、9 组顺序与失败停止、预检失败与运行 notes、子进程非零和中断返回及后代清理。
- NumPy 1.21.6 的 `np.int` 存在但会发弃用警告；兼容检查没有修改 NumPy。模拟设备检查不等于 4090 CUDA、FADC CUDA 扩展的实机运行。真实数据、官方 ImageNet-L 权重、可信 CamVid 初始化权重和训练结果未验证。

## 记录失败与退出状态回归（2026-11-12）

- 执行者：terra-worker（gpt-6.1-terra / xhigh）。局部修复 `tools/train_pidnet_l_4090.py`：最终 notes/report 分别尝试写入，错误输出路径和子进程状态；保留已知非零或信号状态，成功运行但记录失败时返回非零；记录期间保持信号处理，启动回调异常时有界回收进程组。
- 命令：`/home/robot/anaconda3/envs/segformer/bin/python -B test/20261112-pidnet-l-4090-launcher/scripts/check_launcher.py`；最终日志 `logs/check-launcher-record-failure-final.log`，20 项通过（含原 16 项）。使用受控子进程验证退出 7、SIGTERM、退出 0 时记录失败、suite 停止、最终写入时收到信号和回调异常后的后代清理。临时文件位于本目录 `scripts/` 并已清理；未运行训练或评估。
- 最终核对：移除正常路径与 `finally` 中重复的进程组清理后，以同一命令复跑，20 项通过；最终代码对应日志 `logs/check-launcher-record-failure-verified.log`。
