# PIDNet 精度提升报告：源码问题、文献方法与实施顺序

> **仓库**：`/home/robot/catkin_ws/PIDNET`，基线 commit `4c158cf`。
> **调研范围**：2023–2026 年直接改进 PIDNet 的论文，以及同期实时语义分割里可以迁移的模块。
> **本文做了什么**：读源码、核对论文原表、用合成张量验证了几处代码行为。没有训练模型，也没有测过精度或速度。
> **数字口径**：标「文献」的是原论文报告值；标「估计」的是根据文献或代码推断、还没在本仓库验证的数值；只有写「实测」的才是本地代码行为。

## 结论

先说三件事。

第一，当前最值得做的不是换模块，而是把训练和评估链路修干净。边界真值、BAS 空掩码、预训练权重缺 bias、CamVid 训练轮数这四处都是代码级问题。修好它们不需要改推理图，也不会增加线上延迟，但它们决定了后面的消融可不可信。

第二，模型侧有四个明确的瓶颈：PagFM 的通道门控没有启用、1/64 分辨率上的池化窗口过小、主输出停在 1/8 导致 layer1 的 1/4 细节被丢掉、跨分支上采样没有对齐。前两个改动小，后两个更接近论文创新点，代价也更高。

第三，文献里的增益数字不能直接相加。DenseASPP、PMPM、Strip Pooling、FreqFusion、DySample、CWD、重参数化都有各自的原论文结果，但多数不是 PIDNet，或者不是同一个训练配方。把它们当成「候选方向」，不要当成「确定收益」。

建议顺序：先做一份干净的 B0 基线，再用同权重分辨率扫描和边界监督消融判断误差类型，最后才决定动不动结构。具体矩阵在第五节。

## 一、背景：PIDNet 在做什么

这一节给不熟悉分割的读者补基础，熟悉的可以直接跳到第二节。

### 1.1 语义分割和 mIoU

语义分割是给图像里的每个像素分一个类别。Cityscapes 有 19 类（道路、人行道、车、人、天空等），CamVid 有 11 类。模型输出一张和原图同样大的类别图。

评价指标是 mIoU（平均交并比）。对每一类，算预测区域和真实区域的交集除以并集，再把所有类平均。mIoU 78.8 表示平均每类的重叠率是 78.8%。论文里说「涨 1 个点」，通常指 mIoU 增加 1 个百分点。

### 1.2 三个分支

PIDNet 的名字来自 PID 控制器，网络也分成三路：

- **P 分支（比例项）**：一直保持 1/8 分辨率，负责空间细节。它对当前像素的局部差异敏感，像比例项对当前误差快速响应。
- **I 分支（积分项）**：不断下采样到 1/64，感受野很大，负责全局语义。它把整片区域的上下文累积起来，像积分项累积历史误差。
- **D 分支（微分项）**：预测物体边界，控制 P 和 I 的融合。边界是像素值变化最快的地方，像微分项对变化敏感。

这个类比不用太较真，记住「P 管细节、I 管语义、D 管边界」就够了。

### 1.3 一次前向的数据流

输入 1024×1024 的图，先经过 `conv1` 的两个步长卷积降到 1/4（256×256），再过 `layer1`，仍然是 1/4。`layer2` 降到 1/8（128×128），从这里分出三路：

- P 分支在 1/8 上走 `layer3_`、`layer4_`、`layer5_`，中间用 `PagFM` 从 I 分支取两次信息。
- I 分支继续走 `layer3`（1/16）、`layer4`（1/32）、`layer5`（1/64），在 1/64 上做金字塔池化（PAPPM 或 DAPPM），再上采样回 1/8。
- D 分支在 1/8 上走 `layer3_d`、`layer4_d`、`layer5_d`，中间把 I 分支的 1/16、1/32 特征加进来。

最后 `dfm`（Light_Bag 或 Bag）把 P、I、D 融合，`final_layer` 在 1/8 上输出 K 类 logits（Cityscapes 19 类、CamVid 11 类）；评估或推理时再双线性插值回原图尺寸，放大约 8 倍。S/M 用 PAPPM + Light_Bag，L 用 DAPPM + Bag。S→M→L 同时改了容量和路径，所以不能把 L 的增益当成单个模块的消融。

### 1.4 训练时的四个损失

训练时模型返回三个输出：P 分支的辅助语义、融合后的主语义、D 分支的单通道边界。损失由四部分组成：

1. **P 辅助语义损失**，权重 0.4（`configs/` 里的 `BALANCE_WEIGHTS` 是 `[0.4, 1.0]`）。让 P 分支自己也要学会分类，避免它只给主头当特征。
2. **主语义损失**，用 OHEM（只挑最难的像素算损失），权重 1.0。
3. **边界损失**，D 的输出和边缘真值做加权 BCE，权重 20。边界像素很少，所以权重很大。
4. **BAS（boundary-aware supervision）**：只取 D 预测概率大于 0.8 的像素，再算一次主语义损失。目的是让主头在边界附近更准。

前三个是常规做法，第四个是这份代码里比较特殊、也容易出问题的地方。

## 二、源码审计：八个影响精度的问题

下面按「先修什么」排序。每条都给出位置、通俗解释、影响和改法。

### 2.1 边界真值先提边再缩放，void 被当成物体边缘

**位置**：`datasets/base_dataset.py:124-131`、`:154-187`，`utils/criterion.py:107-126`。

**通俗解释**：训练 D 分支需要一个「哪里是边界」的答案图。现在的做法是先在原始标签上做 Canny 边缘检测和 4×4 膨胀，得到一张边缘图，再和图像、标签一起做 0.5–2.1 倍的随机缩放和裁剪。

**为什么影响精度**：

- 边缘宽度会跟着缩放变。缩小 0.5 倍时，4 像素的边变成 2 像素；放大 2.1 倍时变成约 9 像素。D 分支每一步看到的边界粗细不一样，学到的定位会发抖。
- Canny 直接在包含 255（ignore）的标签图上算梯度。类别和 void 的交界也会被当成边缘。比如自车引擎盖、图像黑边、未标注物体的外廓，这些并不是真正的物体边界，但 D 分支会被要求去拟合它们。
- 边界损失没有接收有效像素掩码，void 附近的错误监督会全部算进 loss。

**怎么改**：把提边移到缩放、裁剪、翻转之后，只在「两端都有效且类别不同」的 4 邻域上生成边界，并把有效掩码传给边界损失。注意膨胀会把 void 像素变成正样本，所以屏蔽要在膨胀之后做。当前工作区已经有 `ABLATION.BOUNDARY.TARGET=valid`、`MASK_VOID`、`VOID_RADIUS` 这几个开关，默认值分别是 `legacy`、`False`、`0`，也就是保持原行为；合成验证记录在 `test/20261111-pidnet-ablation-switches/`。

**验证状态**：代码行为已确认；在真实数据上能涨多少还没测。

### 2.2 BAS 辅助损失在空掩码时会报错

**位置**：`utils/utils.py:52-57`，`utils/criterion.py:66-80`。

**通俗解释**：BAS 的想法是「D 分支觉得这里像边界，就让主头在这里再学一遍」。代码用 `sigmoid(D) > 0.8` 选出这些像素，其余像素标成 ignore，再送进主语义损失。

**为什么影响精度**：训练初期 D 分支还很弱，可能一张图里没有任何像素超过 0.8。这时 `bd_label` 全是 ignore，`_ohem_forward` 里的 `pred[min(min_kept, pred.numel()-1)]` 会在空张量上取索引，报 `IndexError`，或者算出 NaN。NaN 一旦进梯度，整个训练就废了。

另外，`torch.where` 的选择本身不可导，所以 BAS 不会通过这个门把梯度传回 D。但 D 仍然通过 Bag 参与主语义梯度，不能说 D 完全没有梯度。还有一个结构问题：最终融合用的是 `layer5_d` 的 4C 特征，辅助头 `seghead_d` 却接在 `layer4_d` 后面，`layer5_d` 没有直接的边界监督。

**怎么改**：空掩码时返回一个和计算图相连的有限零（`score.sum() * 0.0`），不要返回常量 0，否则反向传播会断。是否把硬阈值改成软门控、是否加 Dice，单独做消融，不要和「防 NaN」混在一起记账。

**验证状态**：空掩码的 `IndexError` 和 NaN 已在 CPU 上复现；修复后的行为在实验目录里有合成测试。

### 2.3 预训练权重少了两个 bias

**位置**：`models/pidnet.py:39-46`，`tools/eval.py:64-73`，`tools/custom.py:73-74`。

**通俗解释**：`conv1` 里两个 3×3 卷积后面都紧跟 BatchNorm。卷积的 bias 在这种情况下是多余的：BN 会减均值再缩放，bias 的作用会被抵消。所以 MMSeg 版本的 PIDNet 在导出权重时没有这两个 bias。但本仓库的 `nn.Conv2d` 默认带 bias，加载时这两个张量找不到对应值，就留成了随机初始化。

**为什么影响精度**：随机 bias 会改变第一层特征。实测同一张 2048×1024 的图，只改模型构造的随机种子，主头 argmax 就有 33,967 / 2,097,152 = 1.62% 的像素不同。这不是一个小数。给这两个缺项补零后，纯推理和带辅助头模式的融合输出完全一致。

**怎么改**：把 `conv1` 的两个卷积改成 `bias=False`，或者在加载时显式把缺失的 bias 补零并记录。前者更干净，但会改变参数名和权重格式；后者是格式适配，必须和权重来源一起核对，不能用来掩盖任意缺键。当前工作区的 `utils/checkpoint.py` 已经改成严格加载：缺键、形状不符、前缀不对都会报错，不再静默少载。

**验证状态**：缺键数量、形状、补零后的输出一致性都已实测；原始权重的训练 split 和精度没有核实。

### 2.4 分组学习率没生效，CamVid 训练提前结束

**位置**：`tools/train.py:148-157`、`:177-181`，`utils/utils.py:157-162`。

**通俗解释**：PIDNet 的骨干（P/I/D 的大部分层）通常用较小的学习率，新加的融合头、分类头用较大的学习率。代码里确实写了「如果有两个参数组，第二组用 10 倍学习率」，但 `train.py` 把所有参数塞进了一个组，这段逻辑从来没执行过。

CamVid 还有一行硬编码：`real_end = 121 if 'camvid' in ...`。但学习率日程是按 `END_EPOCH=200` 算的。训练在 121 轮停下时，学习率还剩初始值的 43.36%，模型远没有收敛。

**为什么影响精度**：这两条都会让「基线」偏低。用这样的基线去比较新模块，结论会失真。尤其 CamVid，121 轮和 200 轮的差距可能比很多模块的增益还大。

**怎么改**：把参数按「预训练骨干 / 新初始化层」分成两组，确认每个参数只进一组，记录两组实际学习率。CamVid 要么明确按 121 轮重算日程，要么改到 200 轮；两种配方分开记录，不要混成一条基线。当前工作区还没有默认启用分组学习率，实验时要显式打开。

**验证状态**：代码事实已确认；具体能涨多少取决于数据和训练预算，未测。

### 2.5 PagFM 的门控没有真正打开

**位置**：`models/model_utils.py:268-312`，`models/pidnet.py:65-66`、`:181`、`:193`。

**通俗解释**：PagFM 的作用是「让 P 分支从 I 分支借细节」。它把 P 和 I 的特征各做一次 1×1 投影，算一个相似度图，再按相似度把 I 的特征混进 P。相似度图可以是一张单通道的空间图（每个位置一个权重），也可以是每个通道一张图（每个通道、每个位置一个权重）。

**为什么影响精度**：

- 代码支持 `with_channel=True`，但 `pag3`、`pag4` 实例化时都没传这个参数，实际用的是单通道空间门。所有通道共用同一个权重，表达力弱了一截。
- 相似度是 `sigmoid(sum(x_k * y_q, dim=1))`，没有除以 `sqrt(C_mid)`。两个分支都过 BN，沿通道求和的方差会随通道数变大，sigmoid 容易饱和到 0 或 1，门变成硬开关，梯度回传变差。
- P 分支最后一个 BasicBlock 是 `no_relu=True`，输出无界；I 分支送进 PagFM 前过了 ReLU，之后又过 `compression3/4` 的 Conv+BN。两侧的值域并不一致。互补核查用合成输入验证过，I 侧并不保证非负，所以不能简单说成「激活特征和未激活特征混加」。

**怎么改**：先统计 `pag3`、`pag4` 门控 logits 的分位数、饱和比例，以及边界/内部区域的差异。然后三项分开做：加温度缩放、开通道门、统一激活。不要一次全改，否则不知道是哪一项起作用。

**验证状态**：门控未启用、未缩放是代码事实；饱和程度和对精度的影响需要在真实数据上统计。

### 2.6 1/64 分辨率上的池化窗口太小

**位置**：`models/model_utils.py:114-265`（DAPPM/PAPPM 的 `scale3`、`scale4`）。

**通俗解释**：I 分支到 `layer5` 时，特征图只有输入的 1/64。1024×1024 的输入对应 16×16，960×720 的 CamVid 对应 15×12。PAPPM/DAPPM 想在多个尺度上收集上下文，用了 5×5、9×9、17×17 的平均池化和一个全局池化。

**为什么影响精度**：在 16×16 的图上，17×17 的池化核比整张图还大，窗口里大部分是 padding。它和全局池化（1×1 输出）的区别变小，金字塔的尺度多样性下降。另外，方形池化对车道线、电线杆、护栏这类细长结构不一定合适，容易把周围背景混进来。

但「退化」不能只凭核尺寸判断。互补核查对 16×16 全 1 输入实测，17/8/8 的输出仍有不同的有效覆盖，不能直接说它等价于全局池化。是否值得换，要看真实特征上的响应相关性和下游指标。

**怎么改**：只换 `scale3` 一处做对照，保留 `scale4` 和其余分支。候选有：换成条带池化（水平 H×1、垂直 1×W）、换成 PMPM 的方/横/竖三路池化、或者把核改大并调整 padding。通道数、输出网格、P/D 分支都不动。

**验证状态**：核尺寸和特征图尺寸是代码事实；「退化」程度和对 mIoU 的影响未测。

### 2.7 主输出停在 1/8，layer1 的 1/4 特征被丢掉

**位置**：`models/pidnet.py:169-215`。

**通俗解释**：`conv1` 把图降到 1/4，`layer1` 在 1/4 上提特征，然后 `layer2` 降到 1/8。之后 P、I、D 和 `final_layer` 都在 1/8 或更小的分辨率上工作，最后一次性 8× 双线性插值回原图。

**为什么影响精度**：1/4 特征里有最清晰的边缘和小物体轮廓。它没有跳连到输出，所有细节都要在 1/8 网格上重新恢复。远处的细杆、交通灯、行人的四肢，在 1/8 上可能只占不到一个像素，混叠之后就找不回来了。双线性插值只能把已有的 1/8 预测放大，不能凭空恢复丢掉的结构。

**怎么改**：两条路。便宜的一条是在 1/4 加一个轻量融合头，用 D 分支的边界图控制「只在边界附近注入 1/4 细节」，内部区域保持 1/8 的平滑语义。贵的一条是 D-Guided PointRend：在 1/4 网格上选少量「D 分数高」或「语义不确定」的点，只对这些点提特征、算修正量，再写回粗预测。后者更像论文贡献，但需要证明 D 的选点比只看语义 margin 更有效。

**验证状态**：数据流是代码事实；1/4 细节能找回多少小目标，需要在有标注的数据上测。

### 2.8 跨分支上采样没有对齐

**位置**：`models/pidnet.py:182-206`，`models/model_utils.py:299-309`。

**通俗解释**：I 分支在 1/16、1/32、1/64 上经过步长卷积和池化，特征图上的每个位置对应原图的感受野中心会发生偏移。把它直接双线性放大回 1/8，再和 P/D 相加，深层语义特征在物体边缘处会和浅层特征错开一两个像素。

**为什么影响精度**：错位会让融合后的边界发虚，细长物体（杆、线、栏杆）容易出现断线或重影。这个问题在深层、低分辨率特征上最明显。

**怎么改**：先确认错位是不是主要误差。可以只替换一处上采样做对照：DySample（可学习的点采样）或 FreqFusion（频域低通/高通 + 偏移对齐）。注意它们改的是采样方式，不是整个融合模块；PagFM 的门控、Bag 的融合逻辑要分开对照。模型内部 `algc=False`，配置里 `MODEL.ALIGN_CORNERS=True`，内外标准也不一致，统一 `align_corners` 是另一个独立变量。

**验证状态**：错位是结构上的合理推断；本地没有做过特征对齐误差的量化。

## 三、文献里有哪些方法可以借

这一节只列与本仓库最相关的几项。完整的 31 项调研、每项的出处和限制，见 [方法调研](pidnet_methods_survey/report.md)；代码草案的逐条核查见 [互补核查](pidnet_methods_survey/complementary-review.md)。

### 3.1 直接改 PIDNet 的工作

| 工作 | 改了什么 | 原表结果 | 可信度与限制 |
| --- | --- | --- | --- |
| **Tan / Jin 2025**，MDPI Electronics 14(5):871 | I 末端用 DenseASPP 替换 PPM；Bag 加交叉注意力；多分辨率特征提取 | Cityscapes S 78.6→81.1；其中 DenseASPP 单项 79.0 | 原表消融未标 split；ImageNet 初始化；不是同设重训，不能把 81.1 全归给 DenseASPP |
| **Zhang / Zong / Jiang 2025**，MDPI Electronics 14(6):1109 | I 末端 PMPM：方形、横条、竖条三路池化；P 端 RF-FEM；SUFM 上采样融合 | Cityscapes S 78.3→79.1（PMPM 单项），125→103 FPS；全套 79.8 / 94 FPS | 原表未标 val/test、预训练和 TTA；两个基线来自不同论文，不能拿差值给模块排序 |
| **Chen 等 2024**，CVPR，FADC | 在 bottleneck 里用频率自适应膨胀率、卷积核分量和频带权重 | Cityscapes M-FADC 81.0 val / 80.6 test / 37.7 FPS | 原 M 80.1/80.1/39.8 是引用值，不是同设重训；作者代码只有通用 FADC，没有 PIDNet 配置 |
| **任凤雷等 2026**，《光学精密工程》 | I 端逐级池化 + 门控（EPPM）；P/D 的 MCIAM 沿通道做空洞卷积 | 作者模型 PAPPM 78.7→EPPM 79.2；加 MCIAM 79.2 | 是作者自己的三分支模型，不是原版 PIDNet 重训；不能直接当本仓库增益 |
| **Zhao 等 2026**，Scientific Reports，TBMSNet | I 端 MSAPP 多尺度池化 + DWR 空洞深度卷积 | Cityscapes val PAPPM 80.1→MSAPP 80.5 | 作者模型，代码未公开；CamVid 的 86.1 是作者值，不能和官方 80.1 比 |

另外几项（AWA-PIDNet、飞机部件、PIDFormer 等）只核到题录或摘要，原表和协议不完整，不列入道路场景的迁移候选。它们的价值是提示方向，不是提供可复现的增益。

### 3.2 通用模块与训练方法

| 方法 | 一句话原理 | 原表结果（不是本仓库） | 迁移时注意 |
| --- | --- | --- | --- |
| **DenseASPP**，CVPR 2018 | 多层 3×3 空洞卷积密集连接，扩大感受野 | PIDNet-S 78.6→79.0（Tan / Jin 表 3） | 1/64 上只有 16×16，别照搬高分辨率选的 dilation；保留小率或池化路径 |
| **Strip Pooling**，CVPR 2020 | 水平 H×1 和垂直 1×W 的长条池化，专门抓长条结构 | ADE20K R50：短程 40.50，长程条带 41.14，两者 41.92 | 只换 `scale3`，保留全局分支；先测方形/条带/PMPM 三组 |
| **FreqFusion**，TPAMI 2024 | 频域低通平滑深层特征、高通提浅层边缘，再加偏移对齐 | SegFormer-B1 78.5→80.1；UPerNet-R50 78.8→79.8 | 不是通用 interpolate 替代；先只改一处 PagFM 的 2× 融合 |
| **DySample**，ICCV 2023 | 把上采样变成可学习的点采样，用 `grid_sample` 实现 | ADE20K SegFormer-B1 41.68→43.58；单算子 256×120×120 的 ×2 从 1.6 ms 到 6.2 ms | 只适合一处 2× 采样对照；多尺度替换要逐个验证形状和导出 |
| **GCNet**，CVPR 2025 | 训练时多分支卷积，推理时等价折成单路 3×3 | 重训 PIDNet-S 76.4/128.7 FPS → GCNet-S 77.3/193.3 FPS（A100，无 ImageNet） | 折叠前后要验证输出、padding、导出；训练显存会增加 |
| **RDRNet RPPM**，2024 | 在 PAPPM 训练路径加并行分组卷积，部署时折叠 | RDRNet-S 的 RPPM 76.8，比该网络 PAPPM 高 0.6 | 是不同骨干的消融；只在线性可融合的分支上声称零开销 |
| **SCTNet**，AAAI 2024 | 训练期用 Transformer 语义分支指导 CNN，推理时剥离 | 78.5→80.5；同表 DDRNet-23 79.5→79.6 | 学生本来就有强语义分支时收益可能小；教师只用于训练 |
| **BFMSNet**，IET Image Processing 2026 | 边界特征融合 + BCE/Dice 混合边界损失 + 多尺度上下文 | ResNet18：77.22→77.71（边缘融合）→78.21（混合边界损失）→78.53（加上下文）；test 78.2 | 不同结构里的趋势；比较表有混合分辨率，不能推断 PIDNet 的绝对收益 |
| **CWD**，ICCV 2021 | 对每个类别通道在空间上做 softmax，让学生模仿教师的通道分布 | 原论文在 PSPNet 等网络上验证 | 教师必须同 split、同类序；CamVid 需要 11 类教师或明确的 19→11 转换 |

### 3.3 这些数字不能直接相加

同一个模块在不同骨干、不同初始化、不同训练预算下的增益可以差很多。上表的数字只说明「这个方向有人做出来过」，不说明「本仓库照做就能拿到同样的点」。尤其是：

- PIDNet 原论文表 3 里，S 保持 Light-Bag 时 DAPPM 和 PAPPM 的 mIoU 都是 78.8，但 PAPPM 更快。这说明「换更重的池化」不一定涨精度。
- GCNet 表里的 PIDNet-S 是重训的 76.4，不是作者 ImageNet 初始化的 78.8，两个数不能拼成一条对比。
- MMSeg 的 PIDNet 表用 120k 迭代、`cat_max_ratio`、颜色增强和 SyncBN，和本仓库的配方不是一回事。

## 四、六个可落地的改进方案

下面按「改动位置」分组。每个方案都说明做什么、为什么可能有效、代价和风险。

### 4.1 修边界真值与损失（不增加推理成本）

**做什么**：

1. 把边缘提取移到缩放、裁剪、翻转之后。
2. 只在两端都有效、且类别不同的 4 邻域上标边界。
3. 把有效像素掩码传给边界损失；void 附近可以选择屏蔽整条邻域带，也可以只屏蔽 void 像素，两种口径分开记录。
4. 边界损失从「加权 BCE」扩展为「加权 BCE + Dice」，主语义损失可以再加一个 Lovász-Softmax 项。

**为什么可能有效**：BCE 是逐像素算的，对类别不平衡敏感；Dice 直接看预测和真值的重叠率，二者互补。Lovász-Softmax 是 IoU 的可微代理，直接优化集合级的重叠，不会像逐像素损失那样被大片背景主导。边界真值干净之后，D 分支学到的才是真正的物体轮廓，而不是标注掩码的边缘。

**代价**：推理图不变，零额外延迟。训练时多一个损失项，计算量可以忽略。

**风险**：边界可能变粗或变细，细物体（杆、线）可能被过度膨胀吃掉。先做「旧 Canny + 有效掩码」和「有效邻接边界」两组对照，再看边界 F1 和小实例召回，不要只看总 mIoU。

### 4.2 修训练配方（不增加推理成本）

**做什么**：

1. 把参数分成两组：预训练骨干用基础学习率，新初始化的头用 10 倍学习率。
2. 开启权重 EMA（指数滑动平均）。训练时维护一份权重的滑动平均，验证和导出用这份权重。
3. 在数据增强里加 `cat_max_ratio`，限制单张 crop 里最大类的像素占比，强制稀有类进入视野。
4. CamVid 的训练轮数和学习率日程对齐，不再用 121 轮截断。

**为什么可能有效**：预训练层已经比较成熟，用大学习率会破坏它；新层是随机初始化的，需要更大步长。EMA 相当于把训练轨迹上多个时刻的权重平均起来，通常比最后一个 step 更稳。`cat_max_ratio` 解决的是「稀有类在随机 crop 里几乎不出现」的问题。

**代价**：推理图不变，零额外延迟。EMA 会多一份权重副本，训练显存略增。

**风险**：这些改动会让新结果和旧配方不可比。必须把「复现旧配方」和「新配方」分成两条基线，不能混在一起报告。

### 4.3 知识蒸馏（不增加推理成本）

**做什么**：用一个强教师（PIDNet-L、SegFormer-B5 或 Mask2Former）在训练时指导学生。学生还是原来的 PIDNet-S/M，推理时教师完全去掉。

CWD 的做法是：对每个类别通道，在空间位置上做 softmax，得到「这个通道在哪些位置响应强」的分布，再让学生去拟合教师的分布：

$$
\phi(z_{c,i}) = \frac{\exp(z_{c,i}/T)}{\sum_{j \in \Omega}\exp(z_{c,j}/T)},
\qquad
\mathcal{L}_{\text{CWD}} = \frac{T^2}{K}\sum_{c=1}^{K}\sum_{i \in \Omega}
\phi(z_{c,i}^{T})\log\frac{\phi(z_{c,i}^{T})}{\phi(z_{c,i}^{S})}
$$

其中 $T$ 是温度（常用 4），$\Omega$ 是有效像素，$K$ 是类别数。

**为什么可能有效**：教师的每个类别通道在空间上的响应模式，包含了「这类物体通常出现在哪、和哪些类容易混」的信息。学生只学硬标签，学不到这些。CWD 不需要特征投影层，直接作用在 logits 上，实现简单。

**代价**：推理零成本。训练时需要教师前向，或者在训练前把教师的 logits 缓存下来。

**风险**：教师必须在同一 train split 上训练，不能是 test/trainval 训练过的，否则会泄漏。CamVid 是 11 类，不能直接拿 19 类教师按通道蒸馏；要么找 11 类教师，要么明确做类别映射并验证。教师本身的错误也会被学生学走。

### 4.4 结构重参数化（不增加推理成本）

**做什么**：训练时把一个 3×3 卷积扩展成多条并行分支（3×3 + 1×1 + 1×3 + 3×1 + Identity），推理时利用卷积的线性可加性，把所有分支和 BN 折叠成一个 3×3 卷积。

**为什么可能有效**：多条分支提供了更丰富的梯度路径和更大的有效感受野，训练出来的特征更强；折叠之后，推理计算图和原版完全一样，延迟不变。GCNet 和 RDRNet 都是这个思路。

**代价**：推理零成本。训练时显存和计算量增加，折叠逻辑需要自己实现和验证。

**风险**：折叠必须严格对齐 padding、stride 和 BN 参数；导出 ONNX/TensorRT 后要逐层比对输出。不是所有分支都能线性融合，带非线性或注意力的分支不能直接折。

### 4.5 融合与上下文改造（少量延迟）

**做什么**：

- **EnhancedPagFM**：给相似度加温度缩放（除以 $\sqrt{C_{\text{mid}}}$），开启通道门控，并用可学习的偏移对齐替代直接双线性插值。
- **CA-Bag**：在 Bag/Light_Bag 的融合输出后加一个 Coordinate Attention，沿水平和垂直方向做池化，编码位置信息。参数量增加不到 0.05M。
- **Strip-PAPPM**：把 `scale3` 的 17×17 方形池化换成水平 H×1 和垂直 1×W 的条带池化，再在拼接后加一个 ECA 通道门控。

**为什么可能有效**：PagFM 的门控如果一直是单通道硬开关，P 分支从 I 分支借信息的粒度太粗；温度缩放能让门更平滑，通道门能让每个通道独立决定借多少。条带池化适合道路场景里的长条结构（车道线、护栏、电线杆）。Coordinate Attention 把「在图像哪个位置」的信息补回融合特征。

**代价**：估计增加 1.0–2.5 ms（未实测），参数量增加 0.1–0.3M。

**风险**：这些模块容易过拟合，也容易互相干扰。一次只改一处，保留原模块作对照；先看门控分布和边界指标，再看总 mIoU。

### 4.6 1/4 高分辨率细化（延迟较高，偏研究）

**做什么**：保留 `layer1` 的 1/4 特征，在 1/4 网格上选少量点做局部修正。有两种实现：

1. **1/4 边界门控跳跃头**：把 1/8 的融合特征上采样到 1/4，用 D 分支放大后的边界掩码控制「只在边界附近注入 1/4 浅层特征」，再在 1/4 上输出分类。适合部署，结构简单。
2. **D-Guided PointRend**：在 1/4 网格上算每个点的优先级 $S = 0.5 \cdot s_d + 0.5 \cdot s_u$，其中 $s_d$ 是 D 分支的边界分数，$s_u$ 是主头的不确定性（top1 和 top2 概率的差）。只取 $S$ 最高的 Top-K 个点（例如 10%），用 `point_sample` 取 1/4 浅层特征和粗 logits，过一个小型残差 MLP 预测修正量，再 scatter 回粗预测。

**为什么可能有效**：1/8 网格上，远处细杆、交通灯、行人四肢可能只占不到一个像素，粗预测已经把结构抹平了。1/4 特征里这些结构还在。PointRend 的思路是「只在对的地方花计算」，比全图 1/4 稠密预测便宜。D 分支的边界分数能找回「粗预测已经平滑掉、但图像梯度还在」的位置，这是它和只看语义 margin 的区别。

**代价**：估计增加 2.0–4.5 ms（未实测），参数量增加 0.2–0.4M。稀疏采样在小 GPU 上可能因为 gather/scatter 反而变慢。

**风险**：D 分支可能漏掉被粗图完全吞掉的对象；采样点太稀疏时，修正只覆盖少量像素。必须做等预算对照：随机选点、只看 margin、只看 D、D+margin、全网格轻头。只有 D+margin 在相同采样预算和 p95 延迟下超过 margin-only，才值得继续。

## 五、优先级与实验路线

### 5.1 对比矩阵

下表的增益和开销都是**估计**，不是本仓库实测。证据强度分三档：强（有同模型同任务的对照）、中（有原论文对照但模型或协议不同）、弱（只有结构上的合理推断）。

| 编号 | 改什么 | 推理开销 | 参数量 | 估计 mIoU | 证据 |
| --- | --- | --- | --- | --- | --- |
| M1 | 分组学习率 + EMA + `cat_max_ratio` + CamVid 日程 + `conv1` bias 修正 | 0 | 0 | +0.6 ~ +1.2 | 中 |
| M2 | 有效边界真值 + BCE/Dice + Lovász + 空掩码保护 | 0 | 0 | +0.6 ~ +1.1 | 中 |
| M3 | CWD 通道蒸馏（PIDNet-L / SegFormer-B5 教师） | 0 | 0 | +1.0 ~ +1.8 | 中 |
| M4 | GCNet / RDRNet 式重参数化 BasicBlock | 0 | 0 | +0.8 ~ +1.2 | 中 |
| M5 | EnhancedPagFM + CA-Bag + 对齐采样 | +1.0 ~ 2.5 ms | +0.1 ~ 0.3M | +1.2 ~ +2.0 | 弱 |
| M6 | Strip-PAPPM + ECA 尺度门控 | +0.4 ~ 0.8 ms | +0.05M | +0.5 ~ +0.9 | 弱 |
| M7 | 1/4 边界门控跳跃头或 D-Guided PointRend | +2.0 ~ 4.5 ms | +0.2 ~ 0.4M | +0.8 ~ +1.5 | 弱 |

把 M1–M4 叠加，粗估能到 +1.5 ~ +2.8 个点；把 M1、M2、M5–M7 叠加，粗估 +2.0 ~ +3.5 个点。这些数字只是用来排优先级，不能写进论文或对外报告，除非本仓库真的测出来。

### 5.2 两条路线

**路线 A：保持推理速度不变。** 选 M1 + M2 + M3 + M4。适合嵌入式、车载或需要导出 ONNX/TensorRT 且不能改算子的场景。顺序是：

1. 修数据、损失、权重加载和评估入口，做出 B0。
2. 开分组学习率和 EMA，确认训练稳定。
3. 加 CWD 蒸馏，比较无蒸馏、逐像素 KL、CWD 三种。
4. 把 BasicBlock 换成重参数化版本，验证折叠前后输出一致。

这条路线的改动不增加推理算子；重参数化在导出前折叠回原结构。训练成本会增加，上线前仍要同设备复测，不能假设「结构没变，速度就没变」。

**路线 B：冲精度，允许 10%–20% 的延迟折中。** 选 M1 + M2 + M5 + M6 + M7。适合论文或对精度要求更高的视觉系统。论文可以围绕三个点组织：

1. **跨分支融合的相位错位**：用频域/偏移对齐的 PagFM 和带位置注意力的 Bag，解决深层特征上采样后和浅层对不齐的问题。
2. **1/64 各向同性池化的尺度覆盖不足**：用条带金字塔池化（Strip-PAPPM）建模道路场景的长条结构，同时保留全局分支。
3. **1/8 网格的容量瓶颈**：用 D 分支边界分数和语义不确定性联合选点，把 1/4 浅层细节补回少量关键位置。

这三个点都要有等预算的对照，不能只报「加了模块涨了多少」。尤其是第三点，必须证明 D 的选点比只看 margin 多提供了信息，否则就只是 PointRend 的重复。

### 5.3 评估协议

- **数据划分**：Cityscapes 用 `train.lst` 训练、`val.lst` 选型；方案冻结后才用 `trainval.lst` 重训，`test.lst` 只提交官方评测。CamVid 用 `train.lst` → `val.lst`，最后测 233 张独立 test。test 不参与调参、选 checkpoint 或阈值选择。
- **随机种子**：筛选阶段用 304；入围方案至少跑 304、305、306 三组配对种子，报告每组的差值和均值/离散程度。
- **指标**：mIoU、逐类 IoU、3 px 和 5 px 容差下的边界 F1、小实例召回、p50/p95 延迟、峰值显存。Cityscapes 有实例标注时加官方 iIoU。
- **对照**：所有改动和同一份 B0 比，B0 要锁定 commit、权重哈希、数据名单、输入尺寸、插值方式、优化器步数、BN 策略和硬件。不同变体、不同数据集、不同初始化各自建 B0，不能跨组比较。
- **报告**：区分「本地实测」「文献报告」「待验估计」。文献数字不能写成自己的结果。

## 附录：代码草案

下面三段代码只保留结构，落地时要按当前仓库的接口调整。第二段已经按互补核查的结论修正了零偏移网格。

### A.1 有效边界提取

```python
def extract_valid_edge(label, ignore_label=255, edge_size=4, void_radius=2):
    """在缩放/裁剪后的 label 上生成边界。

    只在两端都有效且类别不同的 4 邻域上标边界；膨胀之后再把
    void 邻域屏蔽掉，避免膨胀把 void 像素变成正样本。
    """
    valid = label != ignore_label
    edge = np.zeros(label.shape, dtype=np.uint8)

    diff_h = (label[1:, :] != label[:-1, :]) & valid[1:, :] & valid[:-1, :]
    diff_w = (label[:, 1:] != label[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[1:, :][diff_h] = 255
    edge[:-1, :][diff_h] = 255
    edge[:, 1:][diff_w] = 255
    edge[:, :-1][diff_w] = 255

    kernel = np.ones((edge_size, edge_size), np.uint8)
    edge = (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.float32)

    # 关键：先膨胀，再屏蔽，顺序不能反
    invalid = ~valid
    if void_radius > 0:
        width = 2 * void_radius + 1
        invalid = cv2.dilate(invalid.astype(np.uint8),
                             np.ones((width, width), np.uint8)).astype(bool)
    edge[invalid] = -1.0  # 在损失里忽略
    return edge
```

### A.2 EnhancedPagFM

```python
class EnhancedPagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=True,
                 with_channel=True, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.scale = mid_channels ** -0.5
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            BatchNorm(mid_channels))
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            BatchNorm(mid_channels))
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, 1, bias=False),
                BatchNorm(in_channels))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
        self.offset_conv = nn.Conv2d(mid_channels * 2, 2, 3, padding=1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def _grid(self, h, w, device, dtype):
        # 像素中心坐标：align_corners=False 时必须这样写
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * 2 / h - 1
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * 2 / w - 1
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    def _sample(self, feat, offset, h, w):
        grid = self._grid(h, w, feat.device, feat.dtype)
        # offset 以输出像素为单位，归一化到 [-1, 1]
        norm = torch.tensor([2.0 / w, 2.0 / h],
                            device=feat.device, dtype=feat.dtype)
        return F.grid_sample(feat, grid + offset.permute(0, 2, 3, 1) * norm,
                             mode='bilinear', padding_mode='border',
                             align_corners=False)

    def forward(self, x, y):
        h, w = x.shape[-2], x.shape[-1]
        if self.after_relu:
            x, y = self.relu(x), self.relu(y)
        x_k = self.f_x(x)
        y_q = self.f_y(y)
        y_q_up = F.interpolate(y_q, size=(h, w), mode='bilinear',
                               align_corners=False)
        offset = self.offset_conv(torch.cat([x_k, y_q_up], dim=1))
        y_q = self._sample(y_q, offset, h, w)
        y = self._sample(y, offset, h, w)
        if self.with_channel:
            sim = torch.sigmoid(self.up(x_k * y_q * self.scale))
        else:
            sim = torch.sigmoid(
                x_k.mul(y_q).sum(dim=1, keepdim=True) * self.scale)
        return (1 - sim) * x + sim * y
```

### A.3 条带池化

```python
class StripPool(nn.Module):
    """水平 H×1 和垂直 1×W 两路池化，输出拼回原分辨率。"""
    def __init__(self, inplanes, branch_planes):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(inplanes, branch_planes, 1, bias=False)
        self.conv_w = nn.Conv2d(inplanes, branch_planes, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        y_h = F.interpolate(self.conv_h(self.pool_h(x)), size=(h, w),
                            mode='bilinear', align_corners=False)
        y_w = F.interpolate(self.conv_w(self.pool_w(x)), size=(h, w),
                            mode='bilinear', align_corners=False)
        return y_h + y_w
```

把 `PAPPM.scale3` 换成 `StripPool`，保留 `scale4` 和其余分支，就是一个最小的条带池化对照。

---

> 修订记录：2026-11-11，ZCode（opencode-go-chat/deepseek-v4.1-flash）重写全文，调整结构、补充通俗解释，并核对源码行号、论文数字与链接。保留原始结论与数据来源，按当前代码和核查结果修正了少量表述。
