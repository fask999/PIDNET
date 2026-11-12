# PIDNet 外部审查核验与修复清单

> **Agents:**
> - Sol-Lead（gpt-6.1-sol / max）— 因果证据复核、工具链探针、报告与集成
> - PIDNet Loss Audit（gpt-6.1-terra / xhigh）— BCE、OHEM 与 BAS 探针
> - PIDNet Model Audit（gpt-6.1-sol / high）— FADC、PMPM、checkpoint 与 HEAD 对照
> - PIDNet BN Audit（gpt-6.1-terra / max）— 历史烟测复现与 BN 配对探针
> - Sol-Supervisor（gpt-6.1-sol / high）— 只读核对范围、顺序与实测证据
> - DeepSeek Writer（opencode-go/deepseek-v4.1-flash / high，原生路由）— 提供文字润色提案
>
> **Date:** 2026-11-11
> **Experiment:** 20261111-external-review-audit

## 结论

在正式做边界消融之前，应先把 void 屏蔽与单类 BCE 策略拆开处理。当前开启屏蔽后，若某个**本地批次**的有效像素全为负例，BCE 会从零变为按全权重计算，放大倍数取决于所用的对照批次和梯度度量。OHEM 漏选有效像素的问题也已复现：该选择规则来自基线，当前空选集由 NaN 改为可反传的零。

DenseASPP 原生严格加载不同率的权重时会静默覆盖率元数据，官方严格加载则在写入前拒绝。在本机的两个 CUDA autocast 用例中，FADC 的前向与反向均成功，附件的 dtype 修补目前没有复现依据。合成验证损失约 7575 已重现；配对实验表明，一步更新与旧 BN 统计共同放大了损失，默认开关的对照也出现同样现象。第二份审查所指的本地 L 权重可以用作部分预训练；评估、预测和恢复各有不同的拒绝条件。下文按实际触发路径给出优先级。本轮未改动运行代码。

## 实验设置与已有结论

基线为 `main` 的 HEAD `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`；审查对象是当时工作区已有的 9 个 tracked 修改及 5 个新增核心模块，还有旧实验的脚本、配置和测试。重新确认 Python 3.7.16、PyTorch 1.13.0、torchvision 0.14.0、CUDA 11.7、GTX 1650 4 GiB。第一阶段 GPU 探针串行，未读取仓库权重；第二阶段只做 CPU 补证，仅读取下文列出的本地 L 权重。两阶段均未下载、安装或使用真实数据。默认模型对照直接在内存载入 HEAD 的模型源码：seed 304、CPU、19 类预测模型、同一 64×96 合成输入，S/M/L 的参数和 buffer 键、张量、预测逐位相同。旧实现的三路输出兼容也有原实验记录；不外推为所有版本和所有输入上的证明。基线 YAML 的 Git 差异为空。

旧实验保留了 28 个预设、Cityscapes 形状的 train/eval/custom、CamVid 形状的 eval、torchvision CUDA 对 MMCV CPU 的小张量前后向等[历史证据](../20261111-pidnet-ablation-switches/report.md#L45)；本轮新增的是下列定向对照。原 loader 改进确实在官方严格评估、预测和恢复入口阻断少载和错配；训练预训练仍刻意支持部分加载，DenseASPP 的原生 API 另见第 4 项。已有测试覆盖这些路径，不等于已覆盖 AMP、CamVid 调色板或真实精度。

## 逐项核验

### 1. 边界 BCE：确认开关耦合，训练影响待测

[`criterion.py:107–155`](../../utils/criterion.py#L107) 先展平整个本地批次（训练时是每个 DataParallel replica），再计数；只要该批有正有负就不会进入单类回退。一张没有边界的图和一张混合图同批时，用的是共同的正例比例。mask 开时先剔除 void，单类权重才取 1；mask 关时全负和全正 BCE 均为零，和 HEAD 一致。单开 `TARGET=valid` 不改变这套单类归约，未屏蔽的无效位置仍可作为负例。全 void 时有连图零；把 void 加到同一组有效像素不改变 mask 开时的 loss 和有效梯度。Dice 独立加和，不纳入下表。

固定 `N=1000` 个有效像素、零 logits、`coeff_bce=20`，全负及全正开 mask 的 BCE 均为 **13.862944**。对比以下混合批次；梯度是对 logits 的导数，负例的导数为正，不能等同于模型参数梯度或训练震荡：

| 混合批正像素比例 | 混合 BCE | 全负/混合：BCE 与梯度 L1 | 全负/混合：逐负像素梯度 | 全负/混合：梯度 L2 |
| --- | ---: | ---: | ---: | ---: |
| 1/1000 | 0.027698 | 500.50× | 1000× | 31.64× |
| 1% | 0.274486 | 50.51× | 100× | 10.05× |
| 2% | 0.543427 | 25.51× | 50× | 7.14× |
| 3% | 0.806823 | 17.18× | 33.33× | 5.86× |

混合时全图平均是 `20·p(1−p)·(正例 BCE 均值+负例 BCE 均值)`；单类开 mask 后是普通均值乘 20。附件的 25–50 倍只能对应特定比例和量度，不能用于任意单正对照。[PIDNet 原论文 §3.1、式 (6)](https://arxiv.org/pdf/2206.02066#page=4) 将 20 列为经验设置，未给出附件推测的选值原因；本轮也未测真实街景边界比例。证据：[loss-run.log:4–34](logs/loss-run.log#L4)、[倍率](logs/loss-run.log#L64)。

### 2. FADC：区分启动特性、扩展条件和版本行为

| 附件子项 | 核验及入口影响 |
| --- | --- |
| 首步零梯度 | **本仓库 FADC 模式的首步行为；初始化设计取舍。** gate 前两支的 project.weight、GroupNorm weight/bias 首轮数据梯度均为 0，gates.weight 首轮已有数据梯度；`wd=5e-4` 时两支 project 只发生衰减，零 norm.bias 不变。第二轮两支 project 的最大绝对梯度为 `6.64e-6` / `1.49e-4`，norm 的梯度也恢复。本样本初态相对 fixed conv 最大差 `2.38e-7`。实测未见这些前层参数持续冻结。 |
| Kaiming 不一致 | **设计取舍。** [PIDNet](../../models/pidnet.py#L110) 对已有卷积做 kaiming_normal，随后才插入 FADC；project 保留 [PyTorch 1.13 Conv2d 的默认 kaiming_uniform](https://github.com/pytorch/pytorch/blob/v1.13.0/torch/nn/modules/conv.py#L135)。已复制的原核和刻意置零的 gate 各有不同角色，不需要为统一分布而重置它们。 |
| fork_rng | **条件性扩展风险。** 同形双插入、相同 CPU RNG 起点时，两次构造都保留外部 RNG，低/高支各自的 project 在两插点间逐位相同，原卷积权重不同；推进 RNG 后相应 project 不再相同。当前 PIDNet 仅选择一个 I 路块，官方配置不会在同一模型触发。 |
| CUDA AMP | **本机未复现附件所称故障。** 卷积输出 offset 为 Half，过 softplus 后传给 deform 的 offset 为 FP32。FP32 输入时算子边界 `x/offset/weight/mask=FP32/FP32/FP32/Half`，输出 FP32；上游卷积产生 Half 输入时为 `Half/FP32/FP32/Half`，输出 Half；两者均完成前后向。关闭 autocast，把同一混合张量送入裸算子才报错。[torchvision 0.14.0 的 AutocastCUDA 实现](https://github.com/pytorch/vision/blob/v0.14.0/torchvision/csrc/ops/autocast/deform_conv2d_kernel.cpp#L21) 将入参转 FP32 后分派，再把输出转回输入 dtype；实测与同样转换的裸算子输出逐位相同。官方 train/eval 当前未启用 AMP；CPU/CUDA FP32 也已实测。此结果不覆盖其他版本、CPU autocast、部署或完整 AMP 训练。 |

FADC 及上述行为均在新增模式，默认关闭。代码：[fadc.py](../../models/fadc.py#L79)、[i_atrous.py](../../models/i_atrous.py#L8)；[实测](logs/model-run.log#L51)。

### 3. PMPM：确认采样格点，精度影响未知

[MixedPool](../../models/context.py#L17) 三路输出对齐；水平条带仅在列轴平均，垂直轴取第 `0,s,2s,…` 行，垂直条带同理取列。冲激探针在 `s=2/4/8` 分别观测到该规律，含尺寸不整除和 1×1 的输入。所谓丢 3/4 或 7/8 适用于可整除尺寸下的单条带正交轴；方池化分支和后续通道融合仍在。该局部降采样有混叠风险，但不能据此断言完整模型有明显混叠或验证精度下降。仅 PMPM 模式受影响，属于新增设计；先保留，若有实测动机再对照抗混叠变体并核对尺寸、对齐和延迟。[冲激记录](logs/model-run.log#L108)。

### 4. DenseASPP：确认原生重载错配，官方严格入口已有保护

源率 `(1,2,3,4)` 到目标率 `(2,4,6,8)` 的定向加载：

| 加载方式 | 实际观察 |
| --- | --- |
| 原生 `load_state_dict(strict=True)` | 返回全部键匹配。目标 buffer 变成源率，卷积 dilation 仍是目标率，权重从源复制。 |
| 统一 loader 严格 | 率值不同直接拒绝，目标张量未改变。官方 eval、custom 和 resume 使用这条路径。 |
| 统一 loader 部分加载 | 加载 465/466 项，仅排除率标记；元数据与算子仍为目标率，匹配的 DenseASPP 权重会跨率迁移。该探针输出与原生跨率加载相同，但两者元数据不同。官方入口中，仅训练的预训练加载采用这一语义。 |
| 同率正对照 | 两种严格加载均成功，输出与源逐位相同。 |

I_ATROUS 把配置编码进键名，原生严格加载不同签名确实抛错，**并非原子拒绝**：fixed/FADC 对照在报错前分别已有 78/80 个目标张量值发生变化，统一严格加载改变 0 项。若希望公开支持原生 strict，Dense 用签名键可补报错，但要保留旧 `_dense_rates` 兼容验证；若还承诺拒绝时不写入，应在根加载前统一预检。明确记录部分加载跨率迁移是否允许，不把它宣传成严格恢复。代码：[context.py:64](../../models/context.py#L64)、[pidnet.py:123](../../models/pidnet.py#L123)、[checkpoint.py:44–60](../../utils/checkpoint.py#L44)；[实测](logs/model-run.log#L117)；[PyTorch 1.13 加载实现](https://github.com/pytorch/pytorch/blob/v1.13.0/torch/nn/modules/module.py)。

### 5. OHEM：确认有效监督漏选，源于基线选择规则

[`criterion.py:66–83`](../../utils/criterion.py#L66) 的顺序索引加严格 `pred < threshold`，会排除阈值本身。阈值 0.9 时，单点 `p=0.95` 或全同分 `p=0.95` 无候选；HEAD 的空均值为 NaN，当前为可反传的零。全 ignore 时 HEAD 索引越界，当前也返回零。三点 `[.91,.95,.98]`、`K=5` 实际仅留 2 个；`[.91,.95,.95,.95,.98]`、`K=3` 仅留 1 个。普通非空对照（含阈值下超过 K 的候选）和 HEAD 数值、梯度相同。实际 FullModel 中一个通过 D 门控且预测置信度 0.95 的 BAS 像素进入语义损失，但 BAS 为 0。

默认训练和训练内验证使用 OHEM，主语义和 BAS 都可能触发，后者的稀疏选区尤其直接；独立 eval/custom 不计算这项损失。空集才回退是最小修复，会保留所有旧非空结果，却不保证数量。若契约为至少 `min(K,N)`，对旧选择不足时补足排序最难样本，并保留原本超过 K 的阈值候选；这会有意改变旧非空不足 K 的结果。验证同分、单点、全 ignore 与常规分支，并明确同分规则。[损失实测](logs/loss-run.log#L36)、[候选保底对照](logs/loss-run.log#L70)、[FullModel](../../utils/utils.py#L49)。

### 6. 工具与标签边界

| 子项 | 结论与范围 | 修复和验证要点 |
| --- | --- | --- |
| custom CamVid 调色板 | **确认。** `--cfg` 构建 11 类，类 0 仍渲染为 Cityscapes RGB `(128,64,128)`，而数据集定义为 `(0,128,192)`；影响 PNG 颜色，不影响 argmax。硬编码来自旧入口，CamVid 配置入口是新增。 | 按数据集选择 palette，保留旧 Cityscapes 默认；用 CamVid 类色与 Cityscapes 对照。 |
| custom 模型路径 | **确认。** 不传 `--p` 时，`--cfg` 中的 `TEST.MODEL_FILE` 被忽略，实际用旧相对 Cityscapes 路径；显式 `--p` 生效。独立 eval 已用配置路径。 | 令显式 `--p` 优先，其次配置，再用明确回退或报错；保留无配置旧模式，覆盖三种路径。 |
| custom align_corners | **确认。** `--cfg MODEL.ALIGN_CORNERS=False` 仍用 True 做最终 resize；默认 True 不受影响。旧硬编码未适配新入口，官方 train/eval 的相应 resize 已读配置。 | 最终 resize 在配置模式读该值，旧入口沿用默认；用非恒定 logits 验证 False 与 eval 一致。 |
| wrapper 输出路径 | **确认。** `show` 和实际 yacs 解析证明末尾的固定 `OUTPUT_DIR/LOG_DIR` 覆盖用户参数；属新增实验 wrapper。固定目录还负责实验隔离及 tee 日志。 | 优先拒绝冲突的保留键并说明 `PIDNET_RUN_TAG`；若支持覆盖，校验合法唯一 `test/<exp>` 内路径并统一 wrapper 与工具的目录。仅交换 `"$@"` 会让 tee 与工具日志分开。 |
| `imgnet_pretrained` 形参 | **确认已无效；兼容取舍。** HEAD 用它选择两种权重解析，现在统一 loader，现有 train/eval 仍用该关键字。 | 可注释为兼容参数或按调用方迁移弃用；无需为修功能直接删除。 |
| RESUME 先预训练 | **确认条件性缺陷；HEAD 遗留。** 续训档存在、配置的预训练文件缺失时，实际流程在读取预训练时失败，未尝试恢复。空预训练路径已可跳过。模拟只加 `not RESUME`，缺恢复文件时会到达 epoch 0 训练回调，成为无预训练的新跑。 | 先定位恢复文件；存在时跳过预训练并严格恢复权重和优化器，缺失应明确报错，或显式选择原初始化回退；测三种路径。 |
| RESUME 的 best 指标 | **确认；HEAD 遗留。** 刷新最佳值的当轮先把旧值保存到恢复档，随后的恢复可把较差新结果误判为最佳；下一次保存可追平，并非每轮都固定滞后一轮。模拟旧档 0.35、真实 best 0.45 时，验证 0.40 仍触发 `best.pt` 写入路径。 | 先计算当轮最佳再保存恢复元数据，明确 `best.pt` 与恢复档的保存顺序；回归新高、续跑较低分和无提升。 |
| Cityscapes 负 ignore | **确认条件性缺陷。** 合成 uint8 PNG 解码后，raw 255 经真实转换、缩放和 pad：ignore 255 与 −1 正常，ignore −2 时 raw 255 仍为 255、被视为有效边界，OHEM 报 `Target 255 is out of bounds.`。保留 int32、扩大内部转码条件的内存模拟使该链路通过；优先在 Cityscapes 原始 ID 转换处处理 255。默认 YAML 为 255，CamVid RGB 映射走另一条路。HEAD 也未完整支持负 ignore；当前已修复 −1 的有符号链路，−2 边界仍缺。 | 用 raw 255/0/7、ignore 255/−1/−2 联测 PNG、映射、缩放/pad、边界及 CE。官方 [Cityscapes 标签](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py) 把 license plate 定义为 −1；[生成器](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/json2labelImg.py#L83) 跳过负 ID，不能由此断言官方分发 PNG 都含 255。 |
| valid 边宽 | **确认映射；文档缺口。** 对远离图边和 void 的单一直边，两侧各标一像素再扩张，`WIDTH=1/3/5/7` 实测横截面为 2/4/6/8 像素。默认 valid=3 与 legacy=4 只在这种宽度意义上对齐；靠近 void/边界会截断，不保证每处 `k+1`。只影响新增 valid 分支的参数解读。 | 在边界参数说明标注截面、单位和限制，无需改变算子。 |

代码：[custom.py:42–105](../../tools/custom.py#L42)、[run_ablation.sh:51–67](../20261111-pidnet-ablation-switches/scripts/run_ablation.sh#L51)、[train.py:85、166–208](../../tools/train.py#L166)、[pidnet.py:222–236](../../models/pidnet.py#L222)、[cityscapes.py:45、86–116](../../datasets/cityscapes.py#L86)、[base_dataset.py:55、134–185](../../datasets/base_dataset.py#L134)。[tools-run.log:19–41](logs/tools-run.log#L19) 使用可控 I/O 和训练回调，实跑 CLI 控制流、yacs、PNG 编解码、标签转换与损失；不把它当成实际续训。

## 合成验证损失归因

旧协议是 seed 304、Cityscapes 形状 128×192、PIDNet-S、DenseASPP+FADC rate 2+valid/mask/Dice、无预训练，训练 B=2、SGD `lr=0.01`/`momentum=0.9`/`wd=0.0005` 一步，验证为两图 B=1 各算 loss 再平均。复用原合成生成器、配置副本、数据管线和 FullModel，训练前向复现 **14.690834**（语义 6.061063、边界 4.215402、BAS 4.414369）；按历史验证协议实测 **7575.009277**（语义 3778.671875、边界 331.394302、BAS 3464.943100），[旧日志](../20261111-pidnet-ablation-switches/logs/verify-ablation-post-review.log#L84) 为 7575.002。两者近似复现，不宣称逐位重现。

配对比较固定 B=2，避开全局池化 BN 对样本数的要求，也避免将 B=1 与 B=2 的 loss 归约差混入因果比较。W0 为更新前参数；W1 为更新后参数；“一次统计”来自 W0 的一次训练前向。切 batch BN 时其他模块保持 eval；训练批重估使用冻结的 W1 和一次训练前向，仅改 BN buffers，未使用验证样本估计该组统计。下表均用同一对验证图和标签，logits 数值是插值前主语义头的最大绝对值。

| 权重 / BN 策略 | 总损失 | 语义 | 边界 | BAS | 主语义 logits 最大绝对值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| W0 / 初始 running | 443.558 | 245.107 | 37.421 | 161.030 | 356.32 |
| W0 / 一次统计 running | 146.171 | 77.094 | 16.567 | 52.510 | 135.90 |
| W0 / batch | 14.886 | 6.117 | 3.971 | 4.798 | 12.16 |
| W1 / 初始 running | 12660.436 | 6448.425 | 553.453 | 5658.557 | 10727.12 |
| W1 / 一次统计 running | **7575.033** | 3778.699 | 331.393 | 3464.941 | **5574.43** |
| W1 / batch | **10.965** | 5.155 | 1.978 | 3.831 | 12.15 |
| W1 / 用训练批重估后 running | **16.154** | 7.837 | 2.153 | 6.164 | 23.97 |

固定 W1 和一次统计，交叉互换训练/验证图像及目标，四格总损失为 7518.188–7821.145；切成 batch BN 后为 10.893–11.571。换输入没有解释数量级差。当前代码关闭消融的对照也出现该现象：B=1 均值 276466.000；同 W1、同 B=2 验证图时一次统计 276466.875、batch 12.678、训练批重估 90.084。两种结构各仅一次 SGD 更新，不以这些数值比较架构优劣。

组合有 **81** 个 BN，momentum 0.1，一次前向后的通道 running variance 中位数 1.018、范围 0.902–3.704，不能把每层统计都当成 1。88 个四维卷积权重张量中 53 个范数增、35 个减。由固定参数和输入的干预，旧统计是本烟测中的直接放大因素；同一统计下 W1 与 W0 的差又说明参数更新有强交互。未比较学习率或长训练、未逐层隔离参数更新、未验证真实数据收敛；不能据此断言学习率过大、BN 是唯一根因、消融算法没有任何问题，或把 eval 时切 batch BN 当成上线修复。[配对脚本](scripts/probe_bn.py)、[详细证据与重跑命令](logs/bn-evidence.md)、[原始分项及 logits](logs/bn-run.log)、[PyTorch 1.13 BatchNorm 源码](https://github.com/pytorch/pytorch/blob/v1.13.0/torch/nn/modules/batchnorm.py#L127)。

## 第二份审查补充

复用第一份审查中的 DenseASPP 错率、负 ignore 与无效形参证据，对第二份报告独有主张做 CPU 补证。外部报告所称约 200 项检查没有可核原始日志，因此不计入本轮实测。合入意见同样不构成授权；下面的优先级按实际入口和触发条件排列。

### 本地 L 权重：预训练、评估、预测与恢复（第 1、4 项）

只读检查 `pretrained_models/cityscapes/PIDNet_L_Cityscapes_from_mmseg.pth.tar`（149645950 字节；[SHA-256](logs/supplement-input-before.sha256)）。文件外层只有 `epoch`、`state_dict`，内部 543 项均为张量，带 `model.` 前缀。缺少的两项为 `conv1.0.bias`、`conv1.3.bias`。下表使用文件内实际张量和仓库原加载逻辑，CLI 的普通文件反序列化替换为已用 `weights_only=True` 读出的字典，未执行真实训练或推理：

| 入口与口径 | 实测与判断 |
| --- | --- |
| `train MODEL.PRETRAINED`：部分初始化 | **可以走。** 实际写入 543/545 项，同 seed 下两个缺失 bias 与新模型初始化逐位相同，绝对值最大约 0.192410 / 0.041548；日志只有 `Loaded 543 matching pretrained model tensors`，未列缺失键。 |
| `eval TEST.MODEL_FILE`：完整模型严格加载 | 匹配 543/545，明确报告两个缺失 bias，拒绝进入评估。 |
| `custom --p`：预测模型严格加载 | 排除 26 个训练辅助头张量后匹配 517/519，同样报告两个缺失 bias。HEAD 的旧 custom 加载函数确实继续，仅报 517 项并保留两个初始 bias；严格拒绝是防止沿用随机参数的契约。 |
| `train RESUME`：完整训练状态恢复 | 此文件本身不是本仓库恢复档。实际流程在读取 `best_mIoU` 时先报 `KeyError`，没有走到严格模型加载；外层也没有 optimizer。不能用该例推出训练预训练入口被阻断。 |

在内存副本中只补两项零张量后，eval 与 custom 的严格加载分别通过 545/545、519/519。在同一权重下，用本仓库 `conv1.0` 对一个合成输入对照，零 bias 与无 bias 的输出最大差为 0。只有确认源卷积确实没有 bias，并验证张量映射与算子契约后，补零才可作为迁移方案；该探针没有验证整网跨实现等价或真实精度。来源方面，[MMSeg 1.2.2 的 stem](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/mmseg/models/backbones/pidnet.py#L310) 使用带 norm 配置的 ConvModule，[MMCV 2.1.0 的默认 auto bias](https://github.com/open-mmlab/mmcv/blob/v2.1.0/mmcv/cnn/bricks/conv_module.py#L123) 在配置 norm 时不建 bias；这与文件缺键相容，但仅凭文件名和缺键不能证明其来源或导出版本。本轮未生成转换文件，文件 SHA-256 前后一致。

建议在确实要用该权重做评估或预测前，先验证来源、同输入输出和完整键映射，再选定显式迁移方案，并继续要求完整严格加载。不要用宽松的 `allow_missing` 让评估和预测退回随机 bias。部分预训练可以输出 missing、unexpected、shape mismatch 以及有意跳过项的计数和关键键名，同时保留现有部分初始化语义；恢复仍要求合法训练状态。相关材料：[加载探针](logs/tools-run.log#L45)、[统一 loader](../../utils/checkpoint.py#L14)、[训练入口](../../tools/train.py#L163)。

### 其他追加项

| 条目 | 判断、范围与修复要点 |
| --- | --- |
| 2. DenseASPP 子模块重载 | 原生错率覆写已证实（上文第 4 项）。新增父容器探针中，子 DenseASPP 的公开 `load_state_dict` 在直接调用时触发 guard，但父级严格加载的 guard 调用数为 0，仍覆盖元数据而不改 dilation。[PyTorch 1.13 源码](https://github.com/pytorch/pytorch/blob/v1.13.0/torch/nn/modules/module.py#L1478) 递归调用子模块的 `_load_from_state_dict`。只覆写子模块公开方法不够；若承诺父 PIDNet 原生入口，需在父级写入前预检，或在递归钩子层校验并明确失败时可能已写入其他张量。只支持统一 loader 时，应把边界写进文档。 |
| 3. 直接数据集构造 | Cityscapes 和 CamVid 直接传 `bd_target='valid'` 省略宽度，构造成功且宽度是 4，首次取样才抛正奇数错误；显式 4 同样失败，显式 3 成功。通过配置的 train/eval 使用自动解析的 3，不触发默认冲突。可区分未指定（按目标自动取值）与显式非法 4（构造期报错），不要把显式错误静默改成 3。 |
| 5. Dice 非法目标和通道 | `DICE_WEIGHT=0.5` 与 `MASK_VOID=False` 是可同时解析的独立设置。零 logits、四个 −1 目标时 loss 为 −1.0；正确开 mask 得连图零。clamp 为 0 会得到 0.333333 并反传到原 ignore 像素，改变了监督语义。标准管线把同一 mask 配置传给数据和 loss，因此不应产生该不配对输入；直接调用或自定义数据可触发。边界头约定 `N×1×H×W`，target 是 `N×H×W`；C=2 在两种 mask 下都报形状错误，单改 permute 不会支持多通道。优先校验二值、ignore 与形状契约；在正确 mask 下排除 ignore。 |
| 6. custom 坏图 | 当 `imread` 返回 None，实测 Python 抛 `AttributeError`，异常消息没有文件名；这是旧路径遗留。输入读取失败时带路径报错或明确跳过并汇总。 |
| 7. custom 空目录 | 在 checkpoint、设备初始化成功的受控用例中，空匹配正常退出，0 次读图、0 张预测，只有加载计数，没有空输入提示；旧路径也无保护。尽早提示匹配规则和目录，定义是否用非零状态。 |
| 8. custom `--cfg` 与 `--a` | 两者都可解析；CamVid S 配置配 `--a pidnet-l` 实际构建 S/11 类，无冲突提示。是新增配置入口的优先级问题。区分显式参数与默认值，记录优先级或拒绝显式冲突。 |
| 9. custom `--c False` | 无 `--cfg` 时实测仍构建 19 类，`type=bool` 将非空字符串读为 True；HEAD 遗留。保留既有命令可用性，改用明确的布尔文本解析或数据集参数。 |
| 10. 模型名解析 | S/M/L 的配置校验和训练/预测构造当前一致，但分别依赖子串。实测 `pidnet_l_cityscapes` 被判为 S，`unknown` 落入 L；模型侧根据实际深度二次校验 block，不能纠正模型名误选。属既有模型名解析和新增配置重复规则的维护风险。共用带明确别名的解析器前先核调用方，覆盖 S/M/L、别名、未知名和 block 边界。 |
| 11. FADC CPU Half 异常 | 本机实际 CPU Half 报 `slow_conv2d_cpu … Half`，发生在进入 deform 前（算子调用 0 次），没有被译为缺 DCNv2。在 FP32 前向内人工让 deform 抛同文字，才触发提示，证实异常匹配过宽的条件性风险。保留原始 dtype/后端错误，只翻译明确的扩展不可用情形；不把 CPU Half 诊断与前一份的 CUDA autocast 混为一谈。 |
| 12. `WIDTH=0` | 配置层 0 表示自动：valid→3、legacy→4；旧 [usage](../20261111-pidnet-ablation-switches/usage.md#L10) 已写自动默认，功能没有错。校验消息可说明 0 的含义，并注明 valid 实际边宽与前一份的直边横截面规则。 |
| 13. `np.int` | HEAD 遗留：`utils/utils.py` 混淆矩阵和 train/eval 的结束耗时日志。运行环境 NumPy 1.21.6 实际计算成功并有弃用告警；现有另一 Python 的 NumPy 2.1.3 直接访问该属性报 `AttributeError`，未在其上运行 PIDNet。根据 [NumPy 1.24 发布说明](https://numpy.org/doc/1.24/release/1.24.0-notes.html#expired-deprecations)，1.24 起该别名失效；据此推断触达这三处会报错。用 `int` 或所需的明确 dtype，并在计划升级前验证混淆矩阵和 CLI 结束路径。 |
| 14. 负 ignore | 与前一份第 6 项重复：Cityscapes 原始 255 经有符号管线在 ignore −2 时仍为 255；默认 255 与 −1 的对照正常。复用首次样本，不再重复测。 |
| 15. `imgnet_pretrained` | 与前一份第 6 项重复：已不控制加载；保留形参可兼容原调用。迁移时再弃用，不按缺陷删除。 |

证据：[追加日志](logs/tools-run.log#L55)、[实际探针](scripts/probe_tools.py)、[数据集构造](../../datasets/cityscapes.py#L16)与 [CamVid](../../datasets/camvid.py#L16)、[配置解析](../../utils/ablation_config.py#L51)、[边界损失](../../utils/criterion.py#L129)、[FADC 异常范围](../../models/fadc.py#L138)、[混淆矩阵](../../utils/utils.py#L132)。

## 修复清单

| 优先级 / 触发 | 建议动作与验收 |
| --- | --- |
| **P0：正式比较边界消融前** | 分离 void 过滤和单类 BCE 策略，保持默认和普通双类归约；若要单类监督，显式配置并定义尺度，同时覆盖全正与全负。以固定有效区域、全 void、多图混批、Dice 开/关回归；正式实验另用相同配方记录单类批频率和分项梯度。不能把 Dice 有梯度假设成所有预设都已开启。 |
| **P1：正式依赖 OHEM/BAS 前** | 明确最低保留数的契约；若需要保底，保留阈值候选并补到 `min(K,N)`，覆盖部分同分以及更多候选。说明与 HEAD 的差别，以同一修复后的基线比较消融。 |
| **P1：原生 checkpoint 公开复用前** | 优先要求统一严格 loader；若要支持原生 strict，加入率预检或签名并迁移旧格式，检查错率、错位置、同率、失败后状态和 partial 的约定。现有官方严格入口无需为此暂停。 |
| **P1：需要用本地 L 权重做评估或预测前** | 确认来源、无 bias 契约及映射，再验证显式迁移，评估和预测保持严格加载；训练预训练继续容许部分加载，并报告缺键和跳过项。不要把文件用作训练状态恢复档。 |
| **P1：续训用于正式实验前** | 提前判定恢复档、消除多余预训练依赖，确保缺档处理可见；先更新 best 再保存其元数据。以已有/缺失恢复档、优化器及 best 覆盖回归。 |
| **P2：按入口启用** | custom 的 palette、权重来源、最终插值一起对齐；wrapper 明确保留键；对要用负 ignore 且来源可能含 255 的 Cityscapes 数据修入口并校验。直接 valid 数据集区分省略值和显式非法宽度；直接 Dice 调用校验 ignore 和形状，修正 custom 坏图、空输入和参数行为。补 valid 边宽文档；NumPy 升级前替换旧别名，扩展模型命名前统一解析。若已经依赖相应输入或输出，优先在使用前修。 |
| **观察，暂不改算子** | 不为本机 AMP 强制 cast、不统一重置 FADC 初始化。多点 FADC 扩展时再设计可复现且各插点独立的初始化；PMPM 有真实数据证据后再做抗混叠对照。合成损失作为诊断，正式训练采用同源权重、相同批与评估协议，记录分项和 BN 状态；不由一步结果调整生产评估方式。 |

## 复现、异常与产物

以下命令均在 `/home/robot/catkin_ws/PIDNET` 下执行。BN 首次运行时默认 fixture 路径尚不存在；复跑应选择本实验内的新路径，不会覆盖首次输入。第一阶段的脚本不需要 checkpoint，两个阶段都不会写入旧实验。复跑前请更换已用过的 replay 编号；CUDA 相关项只在有单卡时运行。

```bash
set -o pipefail
EXP=test/20261111-external-review-audit
PY=/home/robot/anaconda3/envs/segformer/bin/python
OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_loss.py" 2>&1 | tee "$EXP/logs/loss-replay.log"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_model.py" 2>&1 | tee "$EXP/logs/model-replay.log"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_bn.py" \
  --fixture "$EXP/tests/bn-fixture/replay-001" 2>&1 | tee "$EXP/logs/bn-replay-001.log"
OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_tools.py" 2>&1 | tee "$EXP/logs/tools-replay.log"
# 只复跑第二阶段：只读本地 L 权重，不重复第一阶段。
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$PY" -B "$EXP/scripts/probe_tools.py" --supplement \
  2>&1 | tee "$EXP/logs/tools-supplement-replay.log"
```

[计划](plan.md)、[损失证据](logs/loss-evidence.md)、[模型证据](logs/model-evidence.md)、[BN 证据](logs/bn-evidence.md)、[工具日志](logs/tools-run.log) 均位于本实验。探针调试中出现过三处问题：模型探针对 offset dtype 的断言起初不正确，其后的裸算子对照漏还原分组形状，工具探针首轮的 CPU 替身缺少 named_parameters。这些问题均已修正，失败记录予以保留，最终运行成功。BN 的重跑路径选项在 CUDA 实验后补入，只复验 CPU 路径校验，计算协议未改。

[前](logs/protected-before.sha256) / [后](logs/protected-after.sha256) 清单核对了运行源码、基线配置、旧实验和输入文本的 SHA-256，实质文件无差异。完整清单仍存在[三处未跟踪 Python 缓存差异](logs/protected-differences.diff)：camvid、cityscapes 各有一份变化，fadc 新增一份。共享工作区的写入归属尚未查证，因此未清理。仓库的 `data/` 未作为探针输入，也未被写入。第一阶段未读取 `pretrained_models/`；第二阶段只读了上文一份本地 L 权重并核对其哈希，未写入或生成权重文件，未对其他资产逐件做内容哈希。本报告没有真实 mIoU、训练稳定性或部署验证。

修订记录（2026-11-11，Sol-Lead）：采用指定 DeepSeek 原生任务返回的五处文字提案，对照原稿复核事实、数字、结论和限制，澄清两处条件及重跑说明；技术表格、引用和命令均保持原样。[原稿](logs/report-before-polish.md) · [润色提案](logs/polish-proposal.json) · [核验记录](logs/polish-verification.log)。

修订记录（2026-11-11，Sol-Lead）：明确首轮零数据梯度只适用于本仓库 FADC 的 gate 前层，并明确未参与探针的仓库资产目录；未重跑实验，实测结果不变。

修订记录（2026-11-11，Sol-Lead）：追加核验第二份审查的 15 项，保留第一阶段结果；同一 DeepSeek 原生润色单元补润新增三段，lead 对照原稿核对加载计数、来源条件、验证范围及全部数字、链接和命令。[追加原稿](logs/report-before-polish-second.md) · [追加润色提案](logs/polish-proposal-second.json)。
