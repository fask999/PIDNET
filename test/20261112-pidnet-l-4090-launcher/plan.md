# PIDNet-L 单卡 RTX 4090 启动器 — 开发与验收计划

> **Agent:** deepseek-writer（opencode-go/deepseek-v4.1-flash / high）
> **Date:** 2026-11-12
> **Experiment:** 20261112-pidnet-l-4090-launcher
> **Ticket:** 338146450247493f9b622188a8549386；预约 d25b975b0a664ef995c2a7fbd327f8bd；单元 preparation
> **分工:** 本单元只写本文件与 templates/plan.md、templates/report.md 三个文稿；脚本 tools/train_pidnet_l_4090.py 由下一单元实现。

## 目标与假设

新增 tools/train_pidnet_l_4090.py，在单张 RTX 4090 桌面版（不接受 4090D 与笔记本机型）上按 PIDNet-L 既有配方训练。脚本复用旧消融接口与配置，负责 GPU 选择、只读预览、输入与权重预检、逐组实验记录；本轮不启动正式训练，也不对精度、耗时或显存作承诺。

假设：旧 L 消融协议（初始化、日程、seed 304、train.lst → val.lst）足以支撑单卡运行，差异只出现在执行方式与运行记录层。该假设在脚本实现和模拟检查完成前属于待验证事项。

## 基线与输入

- 仓库：/home/robot/catkin_ws/PIDNET；任务开始时记录 HEAD 为 4c158cf24ce432f0a8cb43364fae38d93cee0dc3。
- 规范：AGENTS.md 第 2、3、4、5、6、7、9 节；本计划按第 6、9 节撰写。
- 合同：.super-agent/4090-complex-contract.json（行为 launcher / preflight / artifacts / preview / delivery，保留约束与停止条件）。
- 文档：docs/pidnet-l-ablation.md；旧入口 test/20261112-pidnet-l-ablation/scripts/run_ablation.py（14 预设、协议校验、resolved.yaml 与 SHA-256、独占目录、日志落地方式）。
- 环境证据：conda 环境 segformer，Python 3.7.16、torch 1.13.0、torchvision 0.14.0、numpy 1.21.6；本机 GPU 为 GTX 1650、4096 MiB；真实数据与默认 Cityscapes ImageNet-L 权重未就绪。
- 保全基线：.super-agent/evidence/338146450247493f9b622188a8549386/protected.before.json（332 项 / 258 文件 / 3 链接）；最终复核通过后才可确认其未变。

上述数字来自仓库文件与该任务的环境说明，未在本单元执行命令复测。

## 交付物

| 路径 | 内容 | 归属 |
| --- | --- | --- |
| test/20261112-pidnet-l-4090-launcher/plan.md | 本开发计划 | deepseek-writer |
| test/20261112-pidnet-l-4090-launcher/templates/plan.md | 运行计划模板，供 train_pidnet_l_4090.py 填写 | deepseek-writer（模板） |
| test/20261112-pidnet-l-4090-launcher/templates/report.md | 运行报告模板，同上 | deepseek-writer（模板） |

两个模板必须能被 Python str.format(...) 直接填充：plan 模板只用 {date}、{experiment}、{metadata}、{command} 四个占位符；report 模板可用 {date}、{experiment}、{metadata}、{command}、{observations}、{artifacts} 六个占位符；模板内不得出现其他花括号。

## 改动点与必须实现的行为

本单元不改代码、配置、数据、权重与旧实验；下表是交给下一单元实现的行为清单，也是验收对照。

| 必须实现的行为 | 必须保留的约束 | 对应验证 |
| --- | --- | --- |
| 单卡 RTX 4090 桌面版；先设 CUDA_DEVICE_ORDER=PCI_BUS_ID 与 CUDA_VISIBLE_DEVICES（物理可见序号），再导入旧 runner/torch；子进程只见到逻辑 0；FP32 | 不用 AMP、梯度累积、自动 LR 缩放；不把 4090D 或笔记本判为验收通过 | 模拟 4090 型号判定；遮罩式检查导入顺序（导入前即断言环境变量）；真实 L 配置解析 |
| CLI 约定：--dataset（cityscapes、camvid）、--preset（旧 14 预设，默认 baseline）、--i-suite（与 --preset 互斥，固定 9 组）、--run（YYYYMMDD-slug）、--gpu、--batch-size（默认 2）、--workers、--data-root、--pretrained、--dry-run、--stage、--block | 默认单组 Cityscapes/baseline/batch 2；I 套件顺序固定为 baseline、pappm、pmpm、denseaspp、fixed1、fixed2、fadc、combo-fixed、combo-fadc | 互斥与缺参报错；--help 在不装 torch/yacs 时可用；默认参数与配置一致 |
| --i-suite 串行执行，每组目录 test/<--run>-<preset>，任一组失败立即停止后续 | 不并发占同一 GPU；不覆盖已有目录；不改 batch/epoch 迁就失败 | 串行顺序与失败退出码的可定位检查；旧目录拒绝 |
| --dry-run 只打印计划命令与解析后配置，不写文件，不要求可用 CUDA、数据或权重 | 预览仍运行在装有 torch/yacs 的环境（旧 runner import 即加载）；--help 走惰性导入，不依赖二者 | 预览前后目录哈希不变；无 GPU 条件下预览成功 |
| 预检与记录：Python/torch/CUDA、GPU 型号与显存、git HEAD 与 status、配置 SHA-256、split 名单路径与 SHA-256、权重 profile/绝对路径/SHA-256、seed/batch/实际命令 | 预检失败不启动训练；记录写在 run 目录；不静默安装或下载依赖 | 每项都存在可读来源；缺失项显式标为缺失 |
| 协议：City 1024×1024、484 轮、LR 0.01；Cam 960×720、121 轮、LR 0.005；seed 304；FP32 单卡 | City 19 类 2975/500，Cam 11 类 367/101；训练只用 train.lst、验证只用 val.lst、独立测试集不使用；验证单尺度无 flip | 解析后配置与名单行数核对；测试名单不进入训练或选型路径 |
| 权重：City 默认 pretrained_models/imagenet/PIDNet_L_ImageNet.pth.tar；Cam 必须显式给可信 19 类 City-L 权重 | 共享预训练协议不等于官方完整加载过程，记录里不得如此表述；不改权重文件 | City 缺权重时预检失败；Cam 无 --pretrained 时报错；类别与结构不符时报错 |
| 兼容检查：np.int、FADC CUDA 算子可建、以及按旧 L 逻辑在 CPU 完成一次权重载入检查 | 检查失败按预检失败处理，不进入训练；不因检查失败自动换精度或换算子 | 检查结果逐项写入记录，附错误摘要 |
| 失败与退出：启动失败、训练失败、中断、成功都记录状态、日志路径与耗时；保留退出码 | 不采集或承诺 mIoU、训练时间、显存峰值；不把计划命令写成已执行 | 记录字段齐全；中断时保留原始日志 |

## 数据与划分

- Cityscapes：19 类，data/list/cityscapes/train.lst 2975 行 → val.lst 500 行；test.lst 不参与。
- CamVid：11 类，data/list/camvid/train.lst 367 行 → val.lst 101 行；test.lst 不参与。
- 自定义 --data-root 须提供与仓库同构的名单与图像、标签；名单路径与 SHA-256 记入运行记录。
- 名单行数来自仓库名单文件；本单元未复核 train/val 键交集，列为未决项。

## 运行命令

以下均为计划，本单元未执行。脚本由下一单元实现后，实际命令以脚本打印与 logs 记录为准。

~~~bash
cd /home/robot/catkin_ws/PIDNET
/home/robot/anaconda3/envs/segformer/bin/python -B tools/train_pidnet_l_4090.py --dry-run --dataset cityscapes --preset baseline --run 20261112-pidnet-l-city-baseline
~~~

短测（计划，未执行）：

~~~bash
cd /home/robot/catkin_ws/PIDNET
/home/robot/anaconda3/envs/segformer/bin/python -B test/20261112-pidnet-l-4090-launcher/scripts/check_launcher.py
~~~

短测须覆盖：模拟 4090 判定；无写预览；真实 L 配置解析；遮罩导入顺序；坏输入与旧目录拒绝；I 套件串行与失败即停；记录字段；子进程清理。全部在实验目录内进行，不启动正式训练。

## 评估指标与成功标准

本单元成功标准是文本交付合格：三个文件存在、模板可用 str.format 填充、plan 覆盖上述行为与第 9 节边界。脚本行为验收由下一单元以短测证据承担；运行成功不等同于复验结果，也不产生 mIoU 结论。

## 资源与时间预算

单张 RTX 4090 桌面版、FP32、每卡 batch 默认 2。本机只有 GTX 1650 4 GiB，无法实测 4090 显存与耗时，故不设 batch、轮数之外的资源承诺；OOM、数据缺失、权重不匹配或兼容检查失败即为该次运行的失败边界。

## 可复现性清单（第 9 节）与预审问题

仅列问题，不改代码或配置：

1. split 泄漏：脚本尚未实现，train.lst 与 val.lst 的键交集未实测；test.lst 不参与需由脚本断言，当前只有路径与计划约束。
2. seed：304 须在解析配置与命令中显式出现并可核；尚无实现可对照。
3. GPU/batch：4090 型号识别、显存阈值与 batch 2 的可行性均未验证；本机只能做模拟判定。
4. 配置哈希：旧入口保存 resolved.yaml 与 .sha256；新脚本须复用同一机制，尚未实现。
5. 权重：City 默认 ImageNet-L 文件缺失，SHA-256 无法记录；Cam 的 19 类 City-L 权重路径未提供，类别与结构无法核对。
6. 命令与路径：上面的命令为计划态，未执行；输出须落在 test/<run> 或 test/<run>-<preset>，不得写根目录、output/、log/、/tmp。
7. 数字来源：484/121 轮、LR 0.01/0.005、1024×1024 与 960×720 来自旧配置与文档，未在本机重跑核验。
8. 评估协议：单尺度、无 flip、不用测试集选型须在运行记录中留证据位；mIoU 不在本任务采集范围。
9. 兼容检查：numpy 1.21.6 下 np.int 的触发点、torchvision 0.14.0 是否带可用 deform_conv2d 均未实测。
10. 概览：当前环境缺真实数据与 ImageNet-L 权重，本轮不能给出任何训练、精度或性能结论。

## 风险与回滚

- 只新增本目录文件；若目标文件已存在或已有他人内容，停止并上报，不覆盖。
- 不触碰 models/、utils/、datasets/、tools/train.py、基线配置、数据、权重与其他实验目录。
- 发现范围变化（需要改旧训练配方或核心实现）时交领导裁决，不以脚本任务扩大范围。

## 未决事项

- 脚本实现与短测证据、权重来源与 SHA-256、名单交集核对、4090 实机可用性。
- 文档更新（docs/pidnet-l-4090.md）与最终保全复核由后续单元完成。
