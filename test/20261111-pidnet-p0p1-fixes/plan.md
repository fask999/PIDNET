# PIDNet 已确认 P0/P1 修复与行为回归

> **Agent:** Sol-Lead（gpt-6.1-sol/max）
> **Date:** 2026-11-11
> **Experiment:** 20261111-pidnet-p0p1-fixes

## 目标与基线

按用户授权和 [审计修复表](../20261111-external-review-audit/report.md#L152) 处理五类 P0/P1。基线是 `4c158cf24ce432f0a8cb43364fae38d93cee0dc3` 上已有消融修改的工作树；对比默认模型时另以该 HEAD 的实现作参照。主工单 `a806fe5cd7f045d7b24795f33224aff9`。初始状态和受保护路径校验写入 `logs/`。

| 单元 | 修改与兼容约束 | 行为验收 |
| --- | --- | --- |
| 损失 | `utils/criterion.py`、`utils/ablation_config.py`。单类 BCE 默认零权重，`MASK_VOID` 只过滤 void；单类监督显式配置且定义尺度。OHEM 保持严格阈值和排序，在不足时纳入最低置信度的像素至 `min(K,N)`。保持足额路径、双类公式、Dice/BAS 和空有效区域的既有约定。 | 合成测试：全正、全负、全 void、固定有效区、混批、Dice 组合；OHEM 单点、同分、部分不足、阈值超额、`K>N`、全 ignore、`FullModel` 稀疏 BAS 的实际值与梯度。 |
| 几何与权重 | `models/context.py`、`models/pidnet.py`、`utils/checkpoint.py`、`tools/convert_pidnet_l_stem_bias.py`。原生 Dense/PIDNet 加载预检错率、接受旧版同率签名，严格加载路径保持；统一部分加载可显式跳过率元数据并迁移同形张量，保留计数返回并报告缺失、额外、形状及几何。显式迁移已知本地 PIDNet-L 19 类权重为单独副本；只允许无 bias 假设和已核对的键集。 | 父子错率和失败后参数/率不变、部分和同率、实际卷积 dilation；新副本 strict 545/545（eval）和 519/519（custom），本仓库整网无 bias 与补零对照，源文件哈希不变，manifest 可完整复核。默认 S/M/L、已有消融配置和入口兼容另作集成回归。 |
| 续训 | `tools/train.py`、`utils/training_checkpoint.py`。恢复档优先定位和验证，跳过 ImageNet 预训练；模型严格恢复且拒绝错误训练元数据/SGD 布局。验证到新高才更新 best，写入本轮 best 值；保持非恢复配方和验证频率。 | 新高后续训低分、并列、未验证轮、合法恢复跳过缺失预训练、缺/坏档、SGD 组/动量不匹配；核对实际模型参数和优化器状态。旧档若已写入陈旧 best，无法仅从 `best.pt` 推回历史分数。 |
| 独立审查与报告 | 关联审查工单只收冻结后的源码、调用方、行为测试和结果摘要，当前原始授权同链，GLM 跨契约最多两次。技术稿核实后交指定 DeepSeek 原生润色，核对事实后形成 `report.md`。 | 关联工单当前代码版本独立通过、无未决 P0/P1，报告列实际命令、结果、限制、源哈希、开关和兼容边界。 |

任务特征与初始路由：损失公式和明确行为闭环 → Terra/xhigh；PyTorch 递归加载语义与跨格式窄迁移 → Sol/xhigh；恢复入口/SGD/best 语义 → Sol/high。三个单元各自拥有源码及各自脚本；读其他单元接口，不改其归属文件。技术事实稳定后报告单元用 DeepSeek V4.1 Flash/high。复用 CEO 已绑定的 Sol 监工。

## 数据、环境与运行

验证使用合成张量与只读的 `pretrained_models/cityscapes/PIDNet_L_Cityscapes_from_mmseg.pth.tar`。Cityscapes 图像/标注链接不可用，CamVid 图像目录只有说明文件。不执行真实数据训练、验证或测试；不选 checkpoint 或调参。环境 `/home/robot/anaconda3/envs/segformer/bin/python`，Python 3.7.16，PyTorch 1.13.0、TorchVision 0.14.0、CUDA 11.7，GTX 1650 4 GiB 单卡。用 `-B`，CPU 线程限 2；需要真实 CUDA 算子的测试串行。合成用 seed 304、batch 1–2，预计每组秒到分钟级。

```bash
cd /home/robot/catkin_ws/PIDNET
set -o pipefail
EXP=test/20261111-pidnet-p0p1-fixes
PY=/home/robot/anaconda3/envs/segformer/bin/python
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/check_loss.py" 2>&1 | tee "$EXP/logs/loss.log"
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/check_checkpoint.py" 2>&1 | tee "$EXP/logs/checkpoint.log"
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/check_resume.py" 2>&1 | tee "$EXP/logs/resume.log"
```

迁移 CLI 和默认模型/消融集成的确切复验命令记录在 `report.md`。所有新生成权重、临时数据、脚本与日志仅在本实验目录。若实际运行命令调整，报告给出实用命令和差异。

## 风险与停止条件

不改基线配置、数据、源权重、旧实验或其他已存在变更；不提交、不合并。错率保证限定 Dense/PIDNet 的预检与统一加载，其他 PyTorch 错误不推定原子。模型等价验证限本仓库实现和合成输入；文件名和缺键不能证实来源，不承诺真实精度或框架间等价。保持 P2/P3、FADC/PMPM/AMP/BN 设计。出现超范围或无法兑现指定审查/润色的证据时据实报告，同一实施单元最多两次实际尝试。
