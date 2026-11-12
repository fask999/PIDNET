# PIDNet 外部审查核验计划

> **Agent:** Sol-Lead（gpt-6.1-sol / max，/root/gpt_6_1_sol_pidnet_review_audit_lead_0316897296474183a8db1b2b2b9eee81）
> **Date:** 2026-11-11
> **Experiment:** 20261111-external-review-audit

## 目标与验收

按附件各子项核验事实、触发条件、当前入口和相对基线的影响，校准优先级并给出可回归验证的修复清单。本轮只审查；保持运行源码、基线配置、数据、权重和旧实验原样。附件仅作待核实输入，不从其等级、倍率或根因推断结论。

| 单元 / 对应检查 | 必须观察 | 保留约束及验证 |
| --- | --- | --- |
| loss / loss-audit | 两类、全负、全正、全 void 的 BCE 和 logits 梯度；相同有效像素的 mask 对照、本地 batch 聚合、Dice 不混入 BCE 倍率；OHEM 单像素、同分、部分同分与普通非空分支、BAS 路径、HEAD 对照 | 合成 CPU 输入；用实际当前与基线函数、核数值和梯度、记录非空及基线语义 |
| model / model-audit | DenseASPP 原生 strict、统一 strict/partial、相同率对照及算子状态；PMPM 采样行列；FADC 实际运行 dtype、两步参数梯度/衰减和 RNG | 合成 CPU 及单 GPU；直接测装好的版本、含 full precision 和 autocast，记录前置和算子 dtype；仅实验脚本内探针，不修改实现 |
| tools-cause / tools-cause-audit | 旧合成场景分解 eval 大损失；同输入、同权重配对，仅 BN 策略/统计干预；独立比较更新、输入、BN 与默认结构；逐条核验 custom、命令覆盖、RESUME、best 指标、ignore 标签、边宽 | 固定种子与划分，串行 GPU；保留旧实验并记录重现偏差，不以验证样本校准模型用于指标；工具边界优先只读走查和最小合成探针 |
| report-polish / report-audit | 对照附件逐项分类；每项可执行建议、优先级、核验限度和命令；事实核定后由指定 DeepSeek 润色，lead 核对 | 依赖以上三个单元。保留润色前稿和原样回执；逐条核事实、数字、风险和未决；同模型调用失败如实记；源码、基线配置与旧实验前后 SHA-256 一致 |

## 基线、资源与数据

- `main` 的 HEAD 为 `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`。被审查版本为当前已有工作区改进。保留开始/结束时 Git 状态与文件内容清单，基线比较读取 `git show`。
- 复用旧实验 `test/20261111-pidnet-ablation-switches/` 的配置副本、合成数据生成方法和历史日志，只读。原历史组合为 Cityscapes 形状、PIDNet-S、DenseASPP/FADC rate 2、valid+mask+Dice，无预训练权重，seed 304，2 张训练图一次更新，2 张验证图。
- 可用环境：`/home/robot/anaconda3/envs/segformer/bin/python` 3.7.16，torch 1.13.0、torchvision 0.14.0，CUDA 11.7，GTX 1650（4096 MiB，检查时已用 1074 MiB）。CUDA 探针 model 在前、tools-cause 在后，由 lead 发放单 GPU 使用时段。CPU 探针可并行；每种结构最多一次 SGD 更新，不下载、不装包。
- 真实 train/val/test 数据与任何共享 checkpoint 均不参与测量。所有新输入和输出仅放本目录，Python 用 `-B`。测试集不参与决策；不做真实精度或收敛结论。

## 执行与复现

以下从 `/home/robot/catkin_ws/PIDNET` 执行；一律保留合成日志及精确命令。合成验证若不能逐数重现，保留实际差异及可能原因，不伪造复现。

```bash
set -o pipefail
EXP=test/20261111-external-review-audit
PY=/home/robot/anaconda3/envs/segformer/bin/python
OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_loss.py" 2>&1 | tee "$EXP/logs/loss-run.log"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_model.py" 2>&1 | tee "$EXP/logs/model-run.log"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_bn.py" 2>&1 | tee "$EXP/logs/bn-run.log"
OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_tools.py" 2>&1 | tee "$EXP/logs/tools-run.log"
```

## 归属与停止条件

- Sol-Lead 持有工单编排、`plan.md`、`report.md`、源码保护快照及各单元实测验收；CEO 指定的 Sol-Supervisor 只读观察，复用同一角色。
- 三个探针单元各自拥有不同脚本和日志；`tools-cause` 中 BN 探针交给员工，lead 负责 `probe_tools.py` 及工具证据。任务都不得修改运行代码、回滚已有变更、修改环境或向旧目录写入。
- 有 CUDA/算子缺失先定位，无法验证标限制；超预算停止长任务，不补造通过。正式报告的润色单元已纳入封计划，指定 `deepseek-writer / opencode-go/deepseek-v4.1-flash / high`，先核事实再润色。原生传输已知可能拒绝加密任务；只在真实回执确认后考虑相同模型的宿主路径，保持同一单元及次数，不改账本门禁。

## 修订：第二份外部审查

2026-11-11，Sol-Lead。用户在验收前追加第二份审查，见仓库私有工单输入 `.super-agent/external-review-second-input.md`。沿用原工单、单元与检查，保留第一阶段全部证据，不因重复主张重跑原探针。

- 必须补证：本地 L 权重键、预训练/恢复/评估/预测各自契约及缺键输出；父模块 Dense 加载调用；直接数据集构造宽度；Dice 非法输入和 C=1；custom 参数和坏图/空目录；FADC CPU Half 异常范围；NumPy 兼容及模型名。既有 Dense、负 ignore 和死参数结论复用并交叉引用。
- 第一阶段未读共享权重；追加阶段仅 CPU、只读加载 `pretrained_models/cityscapes/PIDNet_L_Cityscapes_from_mmseg.pth.tar`，保存文件 SHA-256，检查前后相同。仅从 ZIP 归档用 `weights_only=True` 载入张量；不做不安全回退、不生成转换文件或模型副本、不跑真实训练或指标、不使用真实数据。其他工具路径继续使用合成输入与受控 I/O。
- 补证由 Sol-Lead 在已有 `probe_tools.py` 与 `tools-run.log` 留存，监工续用。新内容在合并报告前由原指定 DeepSeek 润色，复核事实，真实记录同一单元次数和当前令牌；不改密封合同或预算。

```bash
set -o pipefail
EXP=test/20261111-external-review-audit
PY=/home/robot/anaconda3/envs/segformer/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_tools.py" --supplement 2>&1 | tee -a "$EXP/logs/tools-run.log"
```
