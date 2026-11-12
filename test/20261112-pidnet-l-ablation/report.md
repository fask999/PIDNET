# PIDNet-L 消融迁移验证报告

> **Agents:**
> - sol-lead（GPT-6.1 Sol / max，原生请求）— 调查、验收、报告与集成
> - sol-worker（GPT-6.1 Sol / high，原生请求）— 配置、入口、权重校验与定向验证
> - sol-supervisor（GPT-6.1 Sol / high，原生请求）— 只读监工
> - deepseek-writer（DeepSeek V4.1 Flash / high，原生请求）— 报告润色、复现性预审
> **Date:** 2026-11-12
> **Experiment:** 20261112-pidnet-l-ablation

## 结论

上下文、I 分支空洞和边界监督现已接入 Cityscapes（19 类）与 CamVid（11 类）的 PIDNet-L 配置及独立 runner。默认 L 仍为 DAPPM + Bag；PMPM 和 DenseASPP 替换 DAPPM，PAPPM 单列对照。固定空洞、FADC 和两个 I 分支组合沿用同一默认边界，边界监督另有独立预设。用户可直接通过 [run_ablation.py](scripts/run_ablation.py) 预览、训练、评估和生成预测。

已完成的验证包括两套真实 PIDNet-L 配置的合成前反向和权重检查，以及 Cityscapes 配置用历史转换权重执行的 CUDA 评估和预测烟测。未做正式训练、真实数据评估或 4090 测速，因此本报告不提供 mIoU、精度提升、显存上限或实际延迟结论。

## 实验设置

- 工作分支为 `main`，对照基线为 Git `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`。实验前工作树已有未提交改动，见[起始状态](logs/git-start.txt)；本轮没有提交。
- L 结构为 m=3、n=4、stem 64、上下文宽度 112、语义头宽度 256。两份配置见 [configs/](configs/)；各配置按 `train.lst` → `val.lst` 使用数据，Cityscapes 为 2975/500，CamVid 为 367/101，图像键和标签键均无交集。参见[名单绝对路径和 SHA-256](logs/dataset-lists.sha256)。未使用 test 调参或评估。CamVid 新配置采用 121 轮日程，匹配旧入口的实际训练轮数。
- 环境为 Python 3.7.16、PyTorch 1.13.0、torchvision 0.14.0、CUDA 11.7、单 GTX 1650（4096 MiB）。各检查固定 seed 304；矩阵检查每批 2 张 64×64 合成图，默认模型对比使用 2 张 32×64 输入；CUDA 实际入口夹具使用 1 张 96×64 图、单尺度且不 flip。训练入口权重夹具源 seed 305、目标 seed 304，并在数据入口截停，未执行优化器步进。配置保留 `CUDNN.DETERMINISTIC: false`，因此 seed 304 不用于承诺 GPU 训练的逐位可复现性。
- Cityscapes 配置默认指向 `pretrained_models/imagenet/PIDNet_L_ImageNet.pth.tar`，本机缺少该文件。CamVid 必须显式指定 19 类 PIDNet-L Cityscapes 权重；加载只转移公共非上下文参数，辅助头、分类层和 FADC 新参数重新初始化。两份配置都会拒绝空路径、缺失文件、S/M 权重或缺少必需参数的初始化权重；输入身份校验不证明来源训练 split。

## 实测结果

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 默认结构回归（CPU，合成输入） | S/M/L 的 19 类增强版与预测版参数/缓冲键、同种子初值和 CPU 输出均与 Git 基线逐项相等 | [legacy](logs/worker-legacy.log) |
| 预设、数据及边界 | 两配置各 14 个预设；I-only 组合不改变边界，边界预设不改变网络；合成 void 条带验证了目标处理和梯度屏蔽 | [protocol](logs/worker-protocol.log)、[名单及资源](logs/environment-data.log) |
| L 完整前反向（合成输入） | CPU 完成 20 项；GTX 1650 上的 FADC 与 FADC 组合覆盖两数据集，共 4 项。梯度均为有限值；stage 3 第 3 号块的 FADC offset/mask 梯度非零 | [CPU](logs/worker-cpu.log)、[CUDA](logs/worker-cuda.log) |
| 权重加载与拒绝路径 | 训练入口的合成 ImageNet 形状夹具加载 202 张量；CamVid 使用合成权重和历史转换实包时各加载 451 个公共张量。错误模型、类别、空路径和错空洞配置均被拒绝 | [训练与严格载入](logs/worker-load.log) |
| 实际 CUDA 入口烟测（历史转换权重 + 合成图） | Cityscapes L 转换包严格载入 eval 545、custom 519 张量；合成验证图可完成评估，custom 输出了 96×64 RGB 图；fixed2 读默认权重及 custom 缺少权重均被拒绝 | [实际 CUDA 入口](logs/cli-cuda.log) |
| runner 与原工作区 | 预览不写入；训练前检查计划、目录和名单；评估与预测复用配置并校验 SHA。起始 234 项文件中，受保护的 137 项均未改变 | [runner](logs/worker-runner.log)、[示例预览](logs/usage-preview.log)、[工作区对比](logs/workspace-preservation.log) |

“完整前反向”使用真实 PIDNet-L 结构与合成输入，不代表真实数据上的精度。历史兼容性烟测使用 `/home/robot/catkin_ws/PIDNET/test/20261111-pidnet-p0p1-fixes/output/pidnet-l-cityscapes-zero-stem-bias.pth.tar`（149628415 字节，SHA-256 `9026f05938c07b8a81ee6da033dbbff439c0e4d9a1f0e58e62a9c5c5fc3699f1`）。转换记录明确补零 `conv1.0.bias` 和 `conv1.3.bias`，原 mmseg 包的训练 split 未核。真实 Cityscapes/CamVid train/val 数据文件目前均缺失，官方 ImageNet-L 实包也未验证；因此不能据本次结果判断真实 mIoU、显存上限或延迟。

## 使用与复现

新的 L 实验直接使用 [run_ablation.py](scripts/run_ablation.py)。`show` 仅预览计划命令和完整解析配置，不写入磁盘；正式 `train`、`eval` 和 `custom` 的用法见 [PIDNet-L 用法](../../docs/pidnet-l-ablation.md)。每次真实训练仍须新建 `test/<YYYYMMDD>-<slug>/`，先写署名的 `plan.md`，并准备好实际数据、同源权重和 GPU。

```bash
cd /home/robot/catkin_ws/PIDNET
PY=/home/robot/anaconda3/envs/segformer/bin/python
KIT=test/20261112-pidnet-l-ablation

# 示例：只预览 Cityscapes PMPM，不写盘。
"$PY" -B "$KIT/scripts/run_ablation.py" show cityscapes pmpm 20261112-pidnet-l-city-pmpm
```

本次迁移检查可按以下命令复现：

```bash
set -e
cd /home/robot/catkin_ws/PIDNET
PY=/home/robot/anaconda3/envs/segformer/bin/python
EXP=test/20261112-pidnet-l-ablation
CONVERTED=test/20261111-pidnet-p0p1-fixes/output/pidnet-l-cityscapes-zero-stem-bias.pth.tar
for suite in legacy protocol cpu cuda runner; do
  "$PY" -B -u "$EXP/scripts/check_l_ablation.py" "$suite"
done
"$PY" -B -u "$EXP/scripts/check_l_ablation.py" load --camvid-checkpoint "$CONVERTED"
"$PY" -B "$EXP/logs/check-cli-cuda.py" "$EXP/configs/pidnet_large_cityscapes.yaml"
```

开始快照到本轮结束的核心差异见[任务源码差异](logs/task-source.diff)及[diff 统计](logs/workspace-preservation.log)。基线配置、仓库现有数据、旧实验和权重均保持原样，没有新增正式 checkpoint。最终版本的独立 GLM 审查结论及版本回执单独归档至私有 `.super-agent/evidence/e81e16b2bdc84d59ad51542dd9c69705/glm-final.json`。

数据、可信预训练和目标 GPU 到位后，按 [PIDNet-L 用法](../../docs/pidnet-l-ablation.md)为每个预设分别创建实验目录，固定 train/val 协议和同源初始化，再比较上下文、固定空洞、FADC 及组合效果。P/D 与 P/I 对齐事项见[待办](../../docs/improvement-backlog.md)。

## 修订记录

- 2026-11-12，sol-lead：调查、验收并撰写原始验证报告，完成集成。
- 2026-11-12，deepseek-writer（DeepSeek V4.1 Flash / high，原生请求）：润色中文措辞与结构，补充 runner 入口和范围限定，并预审复现性清单；事实、数字、命令、验证范围及原作者未改。
- 2026-11-12，sol-lead：对照原稿与日志复核，明确实际 CUDA 入口烟测为 Cityscapes，补名单路径和哈希，复现脚本遇失败即停止。
