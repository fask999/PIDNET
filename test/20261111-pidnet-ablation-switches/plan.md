# PIDNet 前三项改进：开关与行为验证

> **Agent:** sol-lead（gpt-6.1-sol/max）
> **Date:** 2026-11-11
> **Experiment:** 20261111-pidnet-ablation-switches

## 目标与基线

基线是 `4c158cf24ce432f0a8cb43364fae38d93cee0dc3` 的 PIDNet 和原有 `configs/`。交付可独立组合的 I 路上下文、D 路边界监督、I 路空洞率三组开关。全部关闭时 S/M 为 PAPPM、L 为 DAPPM，输出顺序、步幅、原权重键和同权重数值不变。验证是实现验收，不能推断街景精度变化。

## 改动点、约束和检查

| 开关 | 计划行为与归属 | 验收 |
| --- | --- | --- |
| `ABLATION.CONTEXT` 与 `ABLATION.DENSE_RATES` | 模型末端的 `spp` 支持 auto/pappm/dappm/pmpm/denseaspp；原模块不改。新 `models/context.py`：PMPM 按方形、横条和竖条分通道池化，保留 PAPPM 并行路径；DenseASPP 密集连接四层 3×3 空洞卷积及残差映射，1/64 特征默认率 [1,2,3,4]。均为本仓库实现选择。 | S/M/L 张量形状、反传；读取原 HEAD 模型，默认及显式基线键和数值对照；非法值早报。 |
| `ABLATION.BOUNDARY` | 数据集新有效邻接标签在缩放、裁剪、flip 后生成，两端同为有效且类别不同时标注，边宽独立。可独立打开 void/pad mask 和邻域抑制，以及 Dice 系数。保持五元组结构；有效性可通过边界目标的 ignore 哨兵传递。边界与语义 criterion/FullModel 只修复直接相关的空监督非有限情形。 | 非方图、缩放/裁剪/flip/pad、细 void、全忽略及正负缺失；mask、Dice 和目标模式独立生效；普通 CE/OHEM 的 FullModel 可反传。 |
| `ABLATION.I_ATROUS` | `none/fixed/fadc` 只定点改变 I 支 `layer3/4/5` 后续块的 `conv2`，BLOCK 为从 0 数的块号，默认位置 layer5[1]，不改下采样块或 P/D。固定模式 rate=1 等同原卷积。FADC 使用真实的 AdaDR/AdaKern/FreqSelect 与可选 torchvision DCNv2（后端调整见修订记录），仅开启时加载；如无扩展则明确失败。 | 比较选中与未选中层、形状/键/梯度；FADC 真实算子 CUDA 前后向、适配器权重重载及缺依赖时的加载隔离。 |
| 接线与证据 | `utils/ablation_config.py` 扩展 yacs，`models/pidnet.py`、训练/评估/预测统一模型构造，必要的权重读取适配在授权代码目录。原 `configs/`、`data/`、`pretrained_models/`、已有文档和实验只读。配置副本、测试、日志、输出与报告归本实验目录。 | 基线、每个单项和代表组合从真实命令行配置生效；19/11 类、合成训练反传及评估输出；训练到评估和预测权重重载；完整 diff、日志和报告。 |

实现依据：[PMPM §3.4](https://www.mdpi.com/2079-9292/14/6/1109)、[PIDNet DenseASPP §4.2](https://www.mdpi.com/2079-9292/14/5/871)、[原 DenseASPP](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.html)、[FADC 正文](https://arxiv.org/abs/2403.05369) 与 [作者算子/Apache-2.0](https://github.com/Linwei-Chen/FADC/tree/923102e6344d9b90ec427cb15201efd514bf316e)。对应本地调查 `docs/pidnet_methods_survey/direct-cases.md`、`context-cases.md` 和 `complementary-review.md`。论文未公开的 PIDNet 放置/具体超参数记录为本实验选择。

## 数据、命令与预算

数据名单保持原样；Cityscapes 和 CamVid 数据在当前路径未就绪，ImageNet 基线文件也不齐备。仅用固定 seed 304 的合成图像/标签和随机或自生成权重进行短烟测，不使用 train/val/test 的真实图像或选择 checkpoint。正式消融建议 Cityscapes 使用 `train.lst`→`val.lst`，CamVid 使用 `train.lst`→`val.lst`；本次不启动训练。

```bash
cd /home/robot/catkin_ws/PIDNET
EXP=test/20261111-pidnet-ablation-switches
PY=/home/robot/anaconda3/envs/segformer/bin/python
set -o pipefail
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" -B "$EXP/tests/verify_ablation.py" 2>&1 | tee "$EXP/logs/verify-ablation.log"
```

环境预检：Python 3.7.16、PyTorch 1.13.0/CUDA 11.7、torchvision 0.14.0/CUDA 11.7、GTX 1650（4 GiB，开始时约 2.8 GiB 空闲）；NumPy 1.21.6、OpenCV 4.1.2。CPU 为主，小尺寸 GPU batch 2（训练态含池化 BN）；预算几分钟行为检查，0 真实训练 epoch。若 DCNv2 CUDA 内核或显存阻断，记录异常和已完成验证。关闭开关回滚模型与数据路径；不提交、合并、下载数据或大权重。

## 修订记录

- 2026-11-11，sol-lead：初选作者参考的 MMCV。实测本机 MMCV 2.1.0 能导入且有 CPU 前后向，但未编译 CUDA 内核。验证现有 torchvision CUDA 的同组 DCNv2 对照后，以它作为单一可选后端，保留三项自适应机制；不改共享环境、不下载匹配的 97 MB MMCV wheel。增加实际 Cityscapes 形状的合成单步训练→评估→自定义预测，以及 CamVid 形状的合成评估，临时样本和模型由脚本在本目录生成和清理。
- 2026-11-11，sol-lead：GLM 发现空洞配置在重载时可能静默错配。只给启用的模型增加序列化的 mode/stage/block/rate 标记，默认模型原键和值不变；同时修订 FreqSelect 的中性残差增益注释。验证同设置严格重载和输出、错率/位置/模式严格拒绝、部分预训练保留目标模型标记，以及三 CLI 重载。
