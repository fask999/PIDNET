# PIDNet-L 消融迁移与验收计划

> **Agent:** sol-lead（GPT-6.1 Sol，max；原生派遣请求）
> **Date:** 2026-11-12
> **Experiment:** 20261112-pidnet-l-ablation

## 目标与边界

把已有上下文、I 分支固定/自适应空洞和边界监督接到实际 PIDNet-L 的 Cityscapes（19 类）与 CamVid（11 类）实验配置。L 保持 m=3、n=4、planes=64、ppm=112、head=256；关闭开关时为 DAPPM + Bag。PMPM 和 DenseASPP 在 L 上替换 DAPPM；PAPPM 作为单独对照。固定空洞率 1、2 与 FADC 标称率 2 均默认在 I stage 5 的第 1 号（从 0 数）块的 conv2；stage 3/4 也可选 L 特有的第 3 号块。I-only 组合沿用同一默认边界；valid、void mask、Dice 保留单独预设和联合预设。P/D 待办不实施。

不改已实现的模型网络结构、基线配置、仓库中现有的数据与权重文件和历史实验。以 Git 4c158cf24ce432f0a8cb43364fae38d93cee0dc3 的 S/M/L 构造作默认行为对照。此前 20261111-pidnet-ablation-switches 的 runner 选 Cityscapes-M、CamVid-S；旧结果不作为 L 结果。

## 改动与验证

| 必须实现或保留 | 预计位置 | 验收 |
| --- | --- | --- |
| L 配置、独立和 I-only 组合预设、默认行为及 S/M 兼容 | 本目录 configs、scripts | 由真实 CLI 解析验证 L 的层数、通道、19/11 类、DAPPM/Bag、上下文/空洞位置、边界配置；默认参数名、初始化及输出对比 Git 基线。 |
| 明确预训练路径；拒绝空路径、尺寸误选、缺必要共享权重和错配预测配置 | utils/ablation_config.py、utils/checkpoint.py、models/pidnet.py、tools/custom.py（L 配置显式启用时生效） | 真实 L 合法载入；拒绝 M/S、空/缺失、错误类别和错空洞配置；既有 checkpoint 严格读取与 S/M 旧接口保留。网络结构、loss、数据加载器不改。 |
| 训练、评估、配置式 custom 的一致设置与保护 | 本目录脚本、docs/pidnet-l-ablation.md | runner 可预览；新 run 独占且先有 plan.md，输出只在其目录。保存解析配置供 eval/custom 重用。真实 L 小尺寸前/反向及入口接线；区分实际执行与夹具。 |
| 保持数据和对照协议 | 本目录 configs 与仓库名单 | Cityscapes train 2975 / val 500，CamVid train 367 / val 101；查相同键无交集。CamVid 的 END_EPOCH 与当前 121 轮实际训练上限一致，不改旧训练日程逻辑。 |
| 产物可追溯 | 本目录 plan.md、report.md、logs | 验证实录，GLM 对最终版本独立只读审查；事实核实后由 DeepSeek 润色报告并复核。 |

Cityscapes 正式初始权重为 PIDNet-L ImageNet（本地未见文件）；CamVid 为 19 类 PIDNet-L Cityscapes 转移（由运行者显式指定，记录来源）。本机只有 Cityscapes mmseg 原包及历史实验的明确零 stem bias 转换副本；其来源训练 split 未核。副本可作兼容性烟测，不默认用于 Cityscapes 验证集选型，也不覆盖旧文件。转移只取共同模块，保持各组权重来源及训练条件一致；报告列明具体加载范围。

## 命令与资源

在仓库根执行；精确命令、结果及可能差异以 report.md 和 logs 为准。

```bash
cd /home/robot/catkin_ws/PIDNET
EXP=test/20261112-pidnet-l-ablation
PY=/home/robot/anaconda3/envs/segformer/bin/python
git rev-parse HEAD
git status --short
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
"$PY" -B -c 'import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.version.cuda)'
```

本机为单 GTX 1650（4 GiB）、PyTorch 1.13.0 / torchvision 0.14.0 / CUDA 11.7。验证 seed 304，优先 CPU 及小尺寸合成 batch（训练形态每卡或 CPU batch ≥2），对 FADC 检查可用设备算子和有限梯度；不执行正式长训或真实数据评估，不生成 mIoU/FPS。数据与 ImageNet 权重缺失时用合成输入/权重测试明确标注，不能作为精度证据。

## 停止和回滚

遇到缺数据、缺权重、GPU 资源不足或检查失败，记录证据；不下载大文件、不换小模型伪装 L、不改基线。正式每次运行另建 test/YYYYMMDD-slug、先写独立 plan.md；本目录只保存迁移验证证据。若需撤销，仅撤本轮新文件及上述核心文件的本轮差异，保留起始未提交改动。

修订记录：2026-11-12，sol-lead 补记实施归属。sol-worker（GPT-6.1 Sol，high；原生请求）实现两份 L 配置、独立 runner、选择性权重校验和定向检查；sol-lead 核对范围、使用文档、实际 CUDA 入口和保全证据；sol-supervisor 只读观察。

2026-11-12，sol-lead 按 deepseek-writer 的复现性预审补记：[四份名单绝对路径和 SHA-256](logs/dataset-lists.sha256)；配置的 `CUDNN.DETERMINISTIC=false`，固定 seed 不等于 GPU 逐位可复现。实际验证命令见 [report.md](report.md)。
