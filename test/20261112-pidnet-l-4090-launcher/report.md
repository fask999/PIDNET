# PIDNet-L 单卡 RTX 4090 启动器 — 结果报告

> **Agents:**
> - deepseek-writer（opencode-go/deepseek-v4.1-flash）— 开发计划、模板、使用说明、本报告与可复现性预审
> - terra-worker（GPT-6.1 Terra / xhigh）— 启动器实现与短测（同档两次）
> - sol-lead（GPT-6.1 Sol / max）— 核验与独立复跑
> - sol-supervisor（GPT-6.1 Sol / high）— 只读监督
> **Date:** 2026-11-12
> **Experiment:** 20261112-pidnet-l-4090-launcher
> **Ticket:** 338146450247493f9b622188a8549386

## 结论摘要

- `tools/train_pidnet_l_4090.py` 已实现单卡 RTX 4090 启动、只读预览、预检、I 套件串行执行与逐组记录；本机短测 20 项通过，worker 终态与 lead 独立复跑为同一组 20 项，不相加。
- 首轮仅 16 项，发现报告写入失败会掩盖子进程退出码；定向修复后补足为 20 项，并以同一命令复跑通过。
- 未在真实 4090 上训练或评估，无精度、耗时或显存结论；本报告只记录运行与验收事实。
- 332 项保全快照（258 文件 / 71 目录 / 3 链接）的实施结束状态与起点逐字节相同；最终复核须在文档写入后另行执行，本文不预称通过。GLM 独立审查尚未开始。

## 实验设置

- git：分支 `main`，HEAD `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`。工作区不干净，`datasets/`、`models/`、`tools/`、`utils/` 等存在未提交修改；每个 run 的预检会记录 `git status --short` 与 `git diff --stat`。
- 环境：`/home/robot/anaconda3/envs/segformer/bin/python`，Python 3.7.16、torch 1.13.0、torchvision 0.14.0、torch CUDA build 11.7、numpy 1.21.6、cv2 4.1.2、PIL 9.4.0、tqdm 4.65.2。GPU 为 GTX 1650：`nvidia-smi` 4096 MiB，torch 设备属性 3903 MiB。
- 实际执行的命令只有短测与公开入口核验，均为模拟或只读，不含训练命令：

```bash
cd /home/robot/catkin_ws/PIDNET
/home/robot/anaconda3/envs/segformer/bin/python -B test/20261112-pidnet-l-4090-launcher/scripts/check_launcher.py
```

- 配置文件：旧套件 `test/20261112-pidnet-l-ablation/configs/pidnet_large_{cityscapes,camvid}.yaml`，运行副本为各 run 的 `configs/resolved.yaml` 及 `.sha256`。
- 协议值（来自套件配置与预览断言，非训练实测）：seed 304、每卡 batch 2、FP32 无 AMP、Cityscapes 19 类 1024×1024 / 484 轮 / LR 0.01、CamVid 11 类 960×720 / 121 轮 / LR 0.005、验证单尺度无 flip；训练步数未运行，无法给出。

## 结果

本任务不产生精度指标，故不填 mIoU；下表是实际运行与验收结果。

| 验证项 | 结果 | 证据 |
| --- | --- | --- |
| worker 短测（最终代码） | 20/20 通过，4.587 s | `logs/check-launcher-record-failure-verified.log` |
| lead 独立复跑 | 20/20 通过，4.576 s | `.super-agent/evidence/338146450247493f9b622188a8549386/lead-tests.log` |
| 首轮短测 | 16/16 通过 | `logs/check-launcher.log` |
| 公开 CLI `--help` | 仓库外 `-S` 可用，仅标准库 | `lead-entry.log` |
| City I 套件预览 | 9 组顺序与配置正确，`--gpu 17` 不校验设备，无写入 | `lead-entry.log`、`verify_entry.py` |
| CamVid combo-fadc 预览 | 保留 11 类/121 轮与显式 GPU/数据/权重/batch/workers/stage/block，无写入 | `lead-entry.log` |
| 本机 GTX 1650 实跑 | 预检拒绝，未启动训练；plan/notes/report 已生成 | `lead-entry.log` |
| 记录失败退出码 | 子进程 7 → shell 7；子进程 0 但 report 不可写 → shell 1；均停止套件 | `lead-entry.log` |
| 保全快照 | 332 项与起点逐字节一致（`cmp` 相同） | `protected.before.json`、`protected.post-impl.json` |

名单事实（`inputs.json`）：Cityscapes 2975/500（test 1525），CamVid 367/101（test 233），键交集均为 0，名单引用文件存在数均为 0；默认 `pretrained_models/imagenet/PIDNet_L_ImageNet.pth.tar` 不存在。

## 异常与失败

- 首轮 16 项后定位到报告写入失败会丢失子进程退出码；terra-worker 同档第二次定向修复（notes/report 分别尝试写入、stderr 给路径与原始码、0 但记录失败返回 1、记录期间保持信号处理），回归后 20 项通过。
- `np.int` 在 numpy 1.21.6 下可用但发 `DeprecationWarning`；未修改依赖。
- 本机无真实数据、官方 Cityscapes ImageNet-L 权重与可信 CamVid 初始化权重；数据与权重相关用例只覆盖缺失、错误与模拟路径。
- 无 OOM、无训练日志、无精度或耗时数据，因为未启动正式训练。

## 产物路径

- 启动器：`tools/train_pidnet_l_4090.py`
- 使用说明：`docs/pidnet-l-4090.md`
- 计划：`test/20261112-pidnet-l-4090-launcher/plan.md`
- 过程记录：`test/20261112-pidnet-l-4090-launcher/notes.md`
- 短测脚本：`test/20261112-pidnet-l-4090-launcher/scripts/check_launcher.py`
- 日志：`test/20261112-pidnet-l-4090-launcher/logs/`
- 模板：`test/20261112-pidnet-l-4090-launcher/templates/{plan,report}.md`
- 证据：`.super-agent/evidence/338146450247493f9b622188a8549386/`（`inputs.json`、`lead-entry.log`、`lead-tests.log`、`protected.*.json`）；最终结构化结果由该工单证据目录独立归档。

## 下一步

- 完成 GLM 独立只读审查，之后再给审查结论；本报告不预写结果。
- 文档写入后执行最终保全复核，确认 332 项仍与起点一致。
- 真实 4090 桌面机、数据与已核实来源的权重到位后，按 `plan.md` 流程先预览再启动；先跑 FADC CUDA 冒烟。
- 核实 CamVid 初始化权重的来源与训练 split；真实 run 建议记录 checkpoint 大小与哈希。

## 可复现性预审（AGENTS 第 9 节）问题清单

1. **split 泄漏**：名单键交集实测为 0（Cityscapes、CamVid 的 train/val、train/test、val/test），但引用文件均不存在，无法核实际图像或标签；同图不同路径的重复无法排除。真实数据到位后需重查。
2. **seed**：304 显式传入并在预览中断言；但配置保留 `CUDNN.DETERMINISTIC=false`、`BENCHMARK=true`，GPU 非确定性未约束，跨机逐位复现无保证。
3. **GPU/batch**：batch 2 只是配置默认；无 4090 实测显存、耗时与 OOM 边界。本机 GTX 1650 数据不能外推。
4. **命令**：文档命令是示例、未执行；证据中的实际执行只有 help、预览、模拟预检与小子进程。真实命令依赖数据根与权重路径。
5. **路径**：默认 `data/` 的引用文件缺失，City 默认权重缺失，Cam 无默认权重，真实 run 当前不可在本机完成；run 目录与旧套件同构、`saved_config` 已测，但端到端 eval 未测。
6. **数字来源**：2975/500、367/101 及 SHA-256 来自 `inputs.json` 实测；484/121、LR、裁剪尺寸来自套件配置与预览断言；20 项与耗时来自两份日志，不能相加。
7. **权重**：预检只在运行时记录 profile、路径与 SHA-256；本机 `pretrained_models/cityscapes/PIDNet_L_Cityscapes_from_mmseg.pth.tar` 是 mmseg 原包（manifest `source.path`，`source_provenance_verified: false`），`test/20261111-pidnet-p0p1-fixes/output/pidnet-l-cityscapes-zero-stem-bias.pth.tar` 是由它补零 stem bias 的转换产物（manifest `target.path`，来源与训练 split 未核），两者都不能当正式初始化；可信 Cam 权重尚未提供。
8. **评估协议**：val 单尺度、无 flip、无 TTA 来自配置与断言，未实际评估；mIoU 未采集，test split 未使用。
9. **代码状态**：HEAD 固定但工作区脏，新克隆无法复现当前行为；正式 run 前应冻结并记录改动范围。
10. **配置与 checkpoint**：`resolved.yaml` 的 SHA-256 已写入并测过读取；checkpoint 哈希属可选项，报告只列存在性，未生成。

## 未决事项

- GLM 独立审查尚未开始；最终保全校验待文档写入后执行。
- 真实 4090、FADC CUDA、真实数据与官方/可信权重均未就绪，正式训练、评估、精度与耗时全部未产生。
- CamVid 初始化权重来源与训练 split 待核；mmseg 原包及其补零转换产物都不能作为正式初始化证据。

## 修订记录

- deepseek-writer，2026-11-12，初版。
- deepseek-writer，2026-11-12，定点修订权重来源表述：区分 mmseg 原包与补零 stem bias 转换产物，其余数字与结论未改。
