# PIDNet 前三项改进：实现与验证报告

> **Agents:**
> - sol-lead / pidnet_ablation_lead（gpt-6.1-sol/max）— 方案、配置预设、集成实测与报告
> - terra-worker / pidnet_context_integrate（gpt-6.1-terra/max）— 上下文、模型/配置入口和权重重载
> - sol-worker / pidnet_boundary（gpt-6.1-sol/high）— 边界目标、损失和数据集验证
> - sol-worker / pidnet_i_atrous（gpt-6.1-sol/high）— I 路空洞、FADC 与算子验证
> - sol-supervisor / pidnet_ablation_watch（gpt-6.1-sol/high）— 只读核对范围和证据
> - DeepSeek（opencode-go/deepseek-v4.1-flash/high）— 独立会话提供润色提案
> **Date:** 2026-11-11
> **Experiment:** 20261111-pidnet-ablation-switches

实现角色的模型和档位来自派遣配置，宿主未回传实际 provider 遥测；润色来源见修订记录。

## 结论

三组开关已实现，支持单项或组合使用；[使用说明](usage.md) 和两份配置副本提供 14 个预设。默认 S/M/L 与原提交的同 seed 初始化张量、state_dict 键和三路输出逐项相同。真实 CUDA FADC 前后向通过；三组同开时，以合成 19 类样本完成训练一步、权重重载、评估和预测，11 类组合也完成了评估重载。[GLM 定向复核](glm-review.md) 确认两项发现均已修复，无新增发现。本次没有实测街景 mIoU、边界精度或吞吐。

## 实现选择

- **上下文。** `models/context.py` 将 [PMPM §3.4](https://www.mdpi.com/2079-9292/14/6/1109) 的分通道方形池化、横条池化与竖条池化嵌入 PAPPM 的 5、9、17 三个尺度，使用同一采样网格，并复用 1×1 投影；原有局部和全局支路保留。DenseASPP 每层接收原输入和所有更早输出，再通过残差投影融合；实现参考 [PIDNet DenseASPP §4.2](https://www.mdpi.com/2079-9292/14/5/871) 与 [DenseASPP 原论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html)，本地采用适合 I 端 1/64 特征的四层结构，默认率为 1/2/3/4。两者均为本仓库适配。P/D、Bag 和预测顺序未改。
- **边界。** `datasets/base_dataset.py` 在 valid 模式下用增强后的标签生成两端有效且类别不同的四邻接界面，扩展只沿有效域传播；mask 单独编码 void/pad 及配置邻域为 −1。BCE 与 Dice 在 mask 开启时都仅使用有效像素，BAS 同样排除无效点；没有有效监督时返回连图零。默认 Canny 时序和双类平衡 BCE 归约不变，默认单类 BCE 仍为零；开启 mask 后，单类有效目标改用普通 BCE。普通 CE 支持 tensor 或多路输出；OHEM 无候选时仍可有限反传。
- **空洞。** `models/i_atrous.py` 只改 I 路 stage 3/4/5 所选后续块的 `conv2`。fixed 模式修改原卷积的 padding/dilation，rate 1 作为等值对照；FADC 保留原权重键，采用 [论文](https://arxiv.org/abs/2403.05369) 与 [作者固定版本](https://github.com/Linwei-Chen/FADC/blob/923102e6344d9b90ec427cb15201efd514bf316e/FADC_only/conv_custom.py) 的空间 AdaDR、低/高 AdaKern 和 2/4/8 FFT FreqSelect。PIDNet 适配选用零填充、softplus 标称率、零残差频带增益、单位核/掩码初始增益、GroupNorm 与 torchvision DCNv2。原源码版权、修改说明和 [Apache-2.0 许可](../../models/LICENSE-FADC) 随附。替换位置和超参数不是论文 PIDNet 配方的复现。
- **入口。** 独立的 yacs 扩展注册到三个工具入口，旧 YAML 未改。训练预训练允许匹配子集；eval/custom 需完整兼容模型参数，支持原始/包装权重以及常见的 `model.`/`module.` 前缀。启用空洞时会保存模式、stage、block 和率标记；训练恢复、评估和预测会拒绝配置错配，部分预训练保留目标设置。预设为 Cityscapes-M 和 CamVid-S 使用独立 `train.lst`、`val.lst`，日志与产物隔离。

## 环境、命令与结果

基线 `main` / `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`，开始时无 tracked 差异；本次不提交。Python 3.7.16、PyTorch 1.13.0、torchvision 0.14.0/CUDA 11.7、NumPy 1.21.6、OpenCV 4.1.2；单卡 GTX 1650 4 GiB，driver 535.230.02。验证 seed 304；无真实 split 数据或预训练 checkpoint。合成 CLI 输入为 128×192，Cityscapes 形状 train 两张、val 两张、batch 2，跑 1 optimizer step；CamVid 形状 val 两张，eval batch 1。评估无多尺度或 flip；临时合成数据、输出和权重写在本实验 `output/`，并由脚本清理。

```bash
cd /home/robot/catkin_ws/PIDNET
EXP=test/20261111-pidnet-ablation-switches
PY=/home/robot/anaconda3/envs/segformer/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
set -o pipefail
"$PY" -B "$EXP/tests/test_context_integration.py" -v 2>&1 | tee "$EXP/logs/context-post-review.log"
"$PY" -B "$EXP/tests/test_boundary.py" -v 2>&1 | tee "$EXP/logs/boundary-worker.log"
FADC_TEST_DEVICE=cuda "$PY" -B "$EXP/tests/test_atrous.py" -v 2>&1 | tee "$EXP/logs/atrous-post-review.log"
FADC_TEST_DEVICE=cpu "$PY" -B "$EXP/tests/test_atrous.py" -v 2>&1 | tee "$EXP/logs/atrous-cpu-repeat.log"
"$PY" -B "$EXP/tests/verify_ablation.py" 2>&1 | tee "$EXP/logs/verify-ablation-post-review.log"
"$PY" -B "$EXP/scripts/compare_dcnv2.py" 2>&1 | tee "$EXP/logs/compare-dcnv2.log"
```

| 检查项 | 本地实测 | 证据 |
| --- | --- | --- |
| 上下文/基线 | 8 项 CPU 通过；S/M/L、11/19 类、5 种上下文前反传；原 HEAD 与默认和显式原池化的初始化张量与输出逐项等值。测试覆盖同几何固定卷积等值重载、错率/位置/模式拒绝和部分预训练。 | [context-post-review.log](logs/context-post-review.log) |
| 边界 | 10 项 CPU 通过；增强、flip、pad、正负类空集、全 ignore、独立 Dice 与 BAS；默认 Canny、BCE 值/梯度、非空 CE/OHEM 精确对照 HEAD。 | [boundary-worker.log](logs/boundary-worker.log) |
| I 路 | 修复后 7 项 CUDA 通过：固定率选点和脉冲位移，FADC 四频带、输入/权重/自适应参数非零梯度、更新后同率严格重载等值、错率与缺依赖明确失败。修复前同组也有 7 项 CPU 通过。 | [atrous-post-review.log](logs/atrous-post-review.log)、[atrous-worker.log](logs/atrous-worker.log) |
| 三入口 | 28 个预设解析符合独立开关；Cityscapes 形状合成训练 1 步，严格重载 509 个 eval 张量和 483 个预测张量并生成 4 张图；CamVid 形状合成评估严格重载 481 个模型张量，错率 4 的配置明确拒绝率 2 的权重。 | [verify-ablation-post-review.log](logs/verify-ablation-post-review.log)，含实际 argv |
| 算子对照 | 同张量、非整数偏移、分组/变形分组均为 2；torchvision CUDA 相对 MMCV CPU 的前向最大绝对差 `1.91e-6`，五路梯度差最大 `5.01e-6`，通过 `atol=rtol=2e-5`。仅验证此小样本的算子行为。 | [可复现脚本](scripts/compare_dcnv2.py)、[compare-dcnv2.log](logs/compare-dcnv2.log) |

## 异常、差异与后续

本机 MMCV 2.1.0 的 CPU 内核可执行，CUDA 执行报设备实现缺失；用已安装的 torchvision CUDA 内核及上述对照解决，未改共享环境。首轮合成 CLI 测试在训练及评估已成功后误判日志空格；修正检查后完整通过，[原日志](logs/verify-ablation-first.log) 保留。

`git diff --check` 与脚本 `bash -n` 通过；`git diff --exit-code -- configs data pretrained_models docs AGENTS.md agent.md` 无差异。相对 HEAD 的 `git diff --stat` 为 9 个已跟踪源文件，237 行加入、121 行移除；新增代码为 `models/{context,fadc,i_atrous}.py`、`models/LICENSE-FADC`、`utils/{ablation_config,checkpoint}.py`，其余新增文件均位于本实验目录。没有改变基线 YAML 或权重文件。

当前 Cityscapes 软链目标不可访问、CamVid 图像目录为空，示例初始化权重未就绪。现存 L MMSeg 转换权重缺两项 stem bias（见[旧核查](../../docs/pidnet_accuracy_research/report.md)），严格入口会拒绝将它作为完整评估权重。尚未测真实验证集、AMP、部署或真实尺寸延迟；后续先固定同源初始化和统一的验证集协议，分别跑各单项，再按验证集结果决定组合。

## 修订记录

- 2026-11-11，sol-lead：根据 [GLM 审查](glm-review.md) 补上 I 路空洞配置的严格重载检查、纠正 FreqSelect 注释，并更新相应的定向验证和合成入口日志。非空洞模型权重键未变。
- 2026-11-11，sol-lead：核对原稿后采用 DeepSeek 的措辞建议。原生写作任务因宿主加密输入不可读未开始，同一模型和档位在独立明文只读会话中完成提案，未调用工具或改文件。回执：`.super-agent/deepseek-polish-proposal.json`。
