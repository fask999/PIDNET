# PIDNet 已确认 P0/P1 修复 — 结果报告

> **Agents:**
> - Sol-Lead（gpt-6.1-sol/max）— 计划、协调、集成验证与报告技术核对
> - Terra-Worker（gpt-6.1-terra/xhigh）— 边界损失、OHEM 与梯度回归
> - Sol-Worker（gpt-6.1-sol/xhigh）— DenseASPP、模型权重迁移与严格加载回归
> - Sol-Worker（gpt-6.1-sol/high）— 续训、SGD 和 best 保存回归
> - Sol-Supervisor（gpt-6.1-sol/high）— 只读核对范围、阶段证据和风险
> - GLM Reviewer（zai/glm-5.3/high）— 对冻结版本独立只读复审
> - DeepSeek Writer（原生任务指定 opencode-go/deepseek-v4.1-flash/high）— 报告正文润色
>
> **Date:** 2026-11-11  
> **Experiment:** 20261111-pidnet-p0p1-fixes

## 结论摘要

本轮处理外部审计确认的五项 P0/P1。默认 PIDNet-S/M/L 的模型键、同 seed 初始化和三个前向输出均与 HEAD 逐位相同；原有上下文、I 路空洞、边界目标、void 过滤和 Dice 开关仍可独立使用或组合使用。损失边界行为有以下有意变化：开启 void 过滤不再隐式启用单类 BCE；OHEM 在有效阈值候选不足时会补齐。既有 P2/P3 不在本轮范围。

在明确假设源 stem 无 bias 的前提下，本地 PIDNet-L 权重生成了单独副本，eval 和 custom 仍执行严格加载。新的续训档会在最佳分数提升的当轮保存正确的 best 元数据，读取续训档也不再依赖预训练文件。所有实测仅使用合成输入和本地权重，因此不能据此判断真实 mIoU、跨框架等价性或部署效果。

## 实验设置

- 仓库 `/home/robot/catkin_ws/PIDNET`；开工 HEAD `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`，分支 `main`。开工时已有消融改动，默认模型另取该 HEAD 的源码作参照。[开工状态](logs/initial-git-status.txt)、[原始修复清单](../20261111-external-review-audit/report.md#L152)。
- 环境 `/home/robot/anaconda3/envs/segformer/bin/python`，Python 3.7.16、PyTorch 1.13.0、torchvision 0.14.0、CUDA 11.7。CPU 回归 seed 304，线程 2；真实算子和 CLI 烟测串行使用一张 GTX 1650（4 GiB）。
- Cityscapes 图像和标注链接不可用，CamVid 图像目录只有说明文件。CUDA 烟测只读旧实验的 YAML 配置副本：Cityscapes 组合使用 PIDNet-S、DenseASPP/FADC/valid/mask/Dice，128×192、训练 B=2 一步，合成 `train.lst`、`val.lst` 各 2 图；CamVid 组合使用 PIDNet-S、PMPM/fixed/valid/mask/Dice，两张合成 `val.lst` 图，B=1。两者不加载 ImageNet 权重，评估单尺度且不翻转，未使用真实 test split。日志记录了展开后的参数。

## 修复与结果

| 事项 | 修复后的契约 | 实测 |
| --- | --- | --- |
| P0 单类边界 BCE | `ABLATION.BOUNDARY.SINGLE_CLASS_BCE_WEIGHT` 默认 `0.0`；`MASK_VOID` 只过滤 void。显式正值时，单类 BCE 用 `coeff_bce × SINGLE_CLASS_BCE_WEIGHT × BCE 均值`；常规二类归约和 Dice 独立。 | 全正、全负、混批、固定有效区域、遮蔽后全 void、Dice 与梯度通过；默认二类 BCE 值和梯度逐位等同 HEAD。 |
| P1 OHEM / BAS | 保留原 `<` 阈值的全部候选。不足时按原置信度排序补至 `min(K,N)`；全 ignore 为零。 | 单点、同分、部分不足、`K>N`、阈值超额及全 ignore 通过。4 个合格候选、`K=2` 时损失与梯度逐位同 HEAD；稀疏 BAS 只选预期像素，损失 `0.0256466471938`，梯度正确。 |
| P1 DenseASPP 权重几何 | 构造率决定卷积几何。原生 DenseASPP、PIDNet 及递归加载在对应子树写入前拒绝已存在的错率标记；同率旧格式仍可加载。统一严格加载拒绝错率；统一部分加载保留目标率，可迁移同形权重并分类提示。 | 父、子及各自嵌套路径在两种 `strict` 值下错率拒绝且子树状态不变；统一部分加载实测 `489/492`，输出缺失、额外、形状、率及忽略项。 |
| P1 本地 L 权重 | `tools/convert_pidnet_l_stem_bias.py` 要求显式 `--assume-biasless-stem`，只接受映射到默认 PIDNet-L Cityscapes 19 类的窄格式。保留源 543 个张量，只补 `conv1.0.bias`、`conv1.3.bias` 为零。 | 原档两个入口均严格拒绝；副本 eval `545/545`、custom `519/519`。CPU `1×3×64×96` 合成输入下，本仓库无 bias stem 参考与补零的三个 eval 输出和 custom logits 最大绝对差都为 `0.0`。 |
| P1 续训与 best | `TRAIN.RESUME=True` 先定位并检查当前运行目录的 `checkpoint.pth.tar`，合法恢复跳过预训练，模型严格加载并校验 SGD。只有本轮验证得到新高才写 `best.pt`，随后保存带当前最佳值的续训档。 | 真实 CPU SGD 的模型、学习率、动量和下一步逐位对齐。25 种坏档或契约错误均在训练及保存前拒绝。新高 `0.65` 后，较低、并列及未验证轮保留 `best.pt` 原字节。 |

默认对照还逐项比对 S/M/L 的 `479 / 479 / 545` 个状态键和三个输出。保留的旧用例通过 20 项 CPU 检查与 3 项 CUDA FADC 检查；两个数据集共 28 个命令预设解析通过。真实 CLI 合成烟测中，Cityscapes 组合单步训练的档案严格加载到 eval（`509` 项）和 custom（`483` 项，4 张 PNG）；CamVid 组合的 model-only 档严格加载 `481` 项，改变固定率后明确拒绝。真实权重迁移的 eval/custom 验收是在两个入口加载成功后、转 CUDA 前截停；整网数值对照独立在 CPU 执行。

`MASK_VOID=True` 的旧单类 fallback 可在有效目标为二值的同配方中，通过 `ABLATION.BOUNDARY.SINGLE_CLASS_BCE_WEIGHT 1.0` 显式重现。[训练 CLI 参数解析](logs/boundary-config-cli.log)确认两个选项可以同时生效。关闭 mask 时，仍使用整个展平张量的原归约分母；开启后，分母只计筛选后的像素。OHEM 最低保留是本轮默认修复，未新增旧错误行为开关；如需复现旧结果，应使用修复前版本并记录。OHEM 同分时按当前 PyTorch 的排序规则选取，跨设备同分索引是否一致尚未验证。

## 实际命令

```bash
cd /home/robot/catkin_ws/PIDNET
set -o pipefail
EXP=test/20261111-pidnet-p0p1-fixes
PY=/home/robot/anaconda3/envs/segformer/bin/python
SRC=pretrained_models/cityscapes/PIDNet_L_Cityscapes_from_mmseg.pth.tar
OUT="$EXP/output/pidnet-l-cityscapes-zero-stem-bias.pth.tar"

PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/check_loss.py"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B tools/convert_pidnet_l_stem_bias.py \
  --source "$SRC" --output "$OUT" --assume-biasless-stem
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/check_checkpoint.py"
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/check_resume.py"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/check_integration.py" cpu
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/check_integration.py" checkpoint-formats
CUDA_VISIBLE_DEVICES=0 FADC_TEST_DEVICE=cuda OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/check_integration.py" cuda
"$PY" -B "$EXP/scripts/check_scope.py"
```

转换拒绝已有路径；重跑转换时将 `OUT` 改为 `output/` 下的新 `.pth.tar` 文件名，并在 `check_checkpoint.py` 加 `--converted "$OUT"`。无需重新生成副本也可用原命令复验。范围校验脚本仅在内容改变时刷新差异和对照清单。

## 异常与限制

迁移只验证了本仓库模型在已测合成输入上，无 bias stem 与补零后的整网输出一致。源包命名和缺键无法证实导出来源；输出仅包含模型状态，不能作为续训档。DenseASPP 的写入前拒绝只覆盖错率标记及其对应子树，其他 PyTorch 加载错误不保证原子性。

如果历史旧档已将落后于 `best.pt` 的指标写入续训文件，就无法仅凭只含权重的 `best.pt` 还原真实分数；启动旧档的正式续训前，需要确认其 best 元数据可信。旧 SGD 档依赖参数位置，无法识别同形参数置换。`best.pt` 和续训档仍是两个文件，进程若在两次写入之间中断，不在事务恢复保证范围内。CUDA 烟测出现既有 `np.int` 弃用提示，相关 P2 代码未作改动。

## 产物与版本

- 主工单 `a806fe5cd7f045d7b24795f33224aff9`；关联只读 GLM 工单 `b1da5d3229d14ad2bd9bffa6d971332e`。[冻结证据](logs/review-evidence.md) · [本轮运行时代码增量](logs/review-delta.diff) · [增量统计](logs/source-delta-stat.txt)。[当前 `git diff --stat`](logs/final-git-diff-stat.txt) 对 HEAD 含开工前改动，本轮仅上述六个已有及两个新增 Python 文件。
- [转换副本](output/pidnet-l-cityscapes-zero-stem-bias.pth.tar)和 [manifest](output/pidnet-l-cityscapes-zero-stem-bias.manifest.json)。源 SHA-256 前后相同：`4974bdfc56f824337a706d055c66db9f6fa475ae8b3a25db490a1ddf507557e8`；输出 SHA-256：`9026f05938c07b8a81ee6da033dbbff439c0e4d9a1f0e58e62a9c5c5fc3699f1`。manifest 记录全量映射和两项补零。
- [损失](logs/loss.log) · [权重与几何](logs/checkpoint-lead.log) · [续训](logs/resume-lead.log) · [CPU 集成](logs/integration-cpu.log) · [checkpoint 旧用例](logs/integration-checkpoint-formats.log) · [CUDA 集成](logs/integration-cuda.log)。[范围日志](logs/scope.log)和前后哈希核对了 92 个受保护路径，含配置、仓库数据、原权重、旧实验。此次改动尚未提交或合并。

## GLM 独立复审

关联只读工单按本轮原始授权运行 `zai/glm-5.3/high`，回执为 `ef805404546f42ee8b83ac1ebbddd0c8`，耗时 `454.127 s`，使用预算 `1/2`。宿主实际验证了禁写和禁用 MCP，审查过程调用工具 `0` 次。审查返回 `pass`，发现项、未闭环项及未验证项均为空；关联工单已通过 CEO 验收。

负责人对照 [GLM 回执](logs/glm-run-1.json)与 [输入清单](logs/glm-material-1.json)，核对了输入摘要和版本令牌 `1673f83b5a9462c269aa44a060363dfc0824950155301b613ce3c1ba9fe72eb3`，并重新核对 19 个文件的 SHA-256。报告正文不属于代码审查范围。

## 下一步

真实数据恢复后，建议使用相同的预训练来源、训练配方和验证协议比较消融；同时记录单类批频率和各损失分项，再决定是否给单类 BCE 正权重。此次结果仅用于确认修复后的行为和兼容性。

修订记录（2026-11-11，Sol-Lead）：采用指定 DeepSeek 原生任务的 8 段润色，核对事实、数值、命令、验证结果和限制；把下一步保持为建议，并明确整网对照的适用范围。见[技术原稿](logs/report-before-polish.md)、[润色提案](logs/polish-proposal.json)和[核验记录](logs/polish-verification.log)。
