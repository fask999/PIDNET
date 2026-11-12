# PIDNet 的 P、D 分支：源码与精度改进优先级

> **Agents:** Sol Lead（gpt-6.1-sol / max，综合与复核）；P、D 两位 Sol Worker（gpt-6.1-sol / high，分支证据）；DeepSeek Writer（opencode-go/deepseek-v4.1-flash / high，原生润色提案）
> **Date:** 2026-11-11（Asia/Shanghai）
> **Ticket:** f3feb792cec5435daa96cf1ba825abc4

优先试三处彼此独立的改动：**P 的第二次 Pag 前加一处 BEVANet 式细节块，D 的 `diff4` 汇合后加一处语义门控；若确认 P/I 错位，再试 `pag3` 的 2× DySample。** 前两项针对细物体漏分和伪边界，第三项针对空间对齐。PMPM/DenseASPP、I fixed/FADC，以及 D 的 valid/mask/Dice 与 P0/P1 修复均沿用现有配方。下表数字来自作者实验；本轮没有修改源码、训练或评估。

## 12 项候选

表中的箭头只在同一来源内比较指定设置。原表条件不完整处按未核处理；未定位到作者代码并不等于没有发布。题名、固定提交、打开的配置、许可和协议见 [P 证据](p-evidence.md)、[D 证据](d-evidence.md)、[融合证据](fusion-evidence.md)。

| 类型 | 论文、实现 | 作者证据与本仓库接入 |
| --- | --- | --- |
| 直接改 PIDNet：P | [RSM-Optimizer，Zhang 等，2025](https://www.mdpi.com/2079-9292/14/6/1109)，未定位作者代码 | RF-FEM 在 P 内串联卷积和多膨胀率分支。Cityscapes 表 1，PIDNet-S 78.3→78.8，已有 PMPM 时 79.1→79.4；单项 +9.6 GFLOPs，split、预训、TTA 未明。可按论文在 `pag4` 前实现窄版，保持原有融合。 |
| 直接改 PIDNet：融合 | [Tan/Jin，2025](https://www.mdpi.com/2079-9292/14/5/871)，未定位作者代码 | Bag 加 P/I 交叉注意力。Cityscapes 表 3 无单独 cross 行；在 DenseASPP 上 79.0→79.5，在 HRNet+DenseASPP 上 80.5→81.1；split/TTA 未标。完整模型的 81.1 不能归给 Bag；先不换共享骨干，若做融合也不宜在 1/8 全图做注意力。 |
| 直接改 PIDNet：域外 P/D | [唐卡改版，Wu 等，2026](https://doi.org/10.1371/journal.pone.0332164)，[作者实现](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/mmseg/models/backbones/newpidnet.py) | 作者默认 ECA-Pag 是缩放单通道门图再加 P 残差，LGFM 在 P 末端；D 的 PConv3 卷一部分通道。Cityscapes 表 8 的 74.38→75.40 是全套（同 80k/1024² crop，split、预训、TTA 未明）；公开道路配置仍是普通 PIDNet 120k。只借单模块思路，唐卡单项不可外推。 |
| 直接改 PIDNet：域外融合 | [Agri-PIDNet-S，Xin 等，2026](https://doi.org/10.1371/journal.pone.0361992)，[BWRBag](https://github.com/xyl-design/xinyanlin/blob/1b1938e7611189d860d935187d93a27af76caa16/agri_pidnet/model.py#L495-L552) | 四类棉田 test 表 4，88.84→只加 BWR 91.79，同时改了基础融合和窗口；表 5 仅恢复窗口是 90.37→91.79。同一 seed 的域外结果。道路细长轮廓若不连贯，可仅在现有 `dfm` 后试窗口残差。 |
| 道路近邻：P/融合 | [BEVANet，Huang 等，ICIP 2025](https://arxiv.org/pdf/2508.07300v1#page=5)，[EVA](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/models/BEVANet.py#L112-L130) / [BGAF](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/models/FB.py#L45-L71) | Cityscapes 无 ImageNet 的表 4，固定 BA/Bag，普通卷积 77.77→SDLSKA+加法 78.60→CKS 78.86；不是 P 单项。固定 EVA 时换 BGAF 79.27，37.29→32.85 FPS；表 4 未单列 split/TTA。先取一个 P 块，BGAF 三个全通道 3×3，另作对照。 |
| 三分支近邻：P/D | [MCIAM，任凤雷等，2026](https://ope.lightpublishing.cn/zh/article/doi/10.37188/OPE.20263401.0167/)，表 1 伪代码 | 作者三分支取消 Pag、保留 Bag。Cityscapes 表 3 无注意力 78.7→P/D 末端都加 MCIAM 79.2，未拆 P/D、未明 split/预训/TTA。GAP 后沿通道做 1D 空洞卷积；若试，只在 `layer5_` 或 `layer5_d` 后加一处。 |
| 可迁移：D 结构 | [Gated-SCNN，Takikawa 等，ICCV 2019](https://arxiv.org/pdf/1907.05740#page=6)，[作者门控](https://github.com/nv-tlabs/GSCNN/blob/edd16c5d3f346a0ec009cec539fbcaf6d79b5111/my_functionals/GatedSpatialConv.py#L15-L58) | Cityscapes fine val，表 4 的边界 F@5 px 69.8→GCL 70.8→加 dual-task 73.0；表 3 的 72.7→74.3 mIoU 已含 shape stream+dual-task。可独立实现一次 `I4→D` 门控，保留本地损失和融合。 |
| 可迁移：D 算子 | [PiDiNet，Su 等，ICCV 2021](https://arxiv.org/pdf/2108.07009#page=6)，[PDC 算子](https://github.com/hellozhuo/pidinet/blob/d21aa881ed9c628571636fad39acfe1fad517ebd/models/ops.py#L48-L93) | 不是 PIDNet 改版。BSDS500 val 表 3，正常宽度普通卷积→PiDiNet 的 ODS/OIS 0.736/0.751→0.757/0.776，无道路 mIoU。先在 `layer3_d` 内用一处中心差分做对照，监测纹理误报和部署等价核。 |
| 可迁移：边界训练 | [Active Boundary Loss，Wang 等，AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/download/20139/19898)，[作者损失](https://github.com/wangchi95/active-boundary-loss/blob/1511507533ad98f04ea26e3648360a6c1d477d37/abl.py#L31-L198) | Cityscapes val 单尺度，DeepLabV3 表 1：CE 79.5→后 20% 加 ABL 79.6；固定 Lovász 后 80.2→80.5。它把预测语义边界向 GT 对齐，接主 logits，不接单通道 D；需适配 void 和空边界。 |
| 可迁移：P/I 对齐 | [DySample，Liu 等，ICCV 2023](https://arxiv.org/pdf/2308.15085#page=7)，[MIT 算子](https://github.com/tiny-smart/dysample/blob/81a1de5caa95d55a0f5488425fa53ec7ef47f8f0/dysample.py#L20-L82) | ADE20K 表 4，SegFormer-B1 双线性→DySample-S+，mIoU 41.68→43.58、bIoU 27.80→29.93；没有 PIDNet 道路对照。先只换 `pag3` 的 I 采样。 |
| 可迁移：P/I 融合 | [FreqFusion，Chen 等，TPAMI 2024](https://arxiv.org/html/2408.12879v1#S5)，[双输入算子](https://github.com/Linwei-Chen/FreqFusion/blob/3fb0c70637a3c194fb74294d3ce4681958b26241/FreqFusion.py#L112-L277) | Cityscapes val 单尺度表 IV，SegFormer-B1 78.5→80.1、+28.1 GFLOPs；不是 PIDNet 或单个 Pag。对齐、低通与 P 高频增强可以分测；默认关闭偏移重采样，主算子许可未核。 |
| 可迁移：输出修边 | [SegFix，Yuan 等，ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570477.pdf#page=12)，[作者后处理](https://github.com/openseg-group/openseg.pytorch/blob/aefc75517b09068d7131a69420bc5f66cb41f0ee/scripts/cityscapes/segfix.py#L59-L95) | 表 6 Cityscapes val，多尺度 Gated-SCNN 81.0→81.5、边界 F 61.4→63.1。固定 PIDNet 主头后可作输出级对照；另需独立边界/方向预测，会增加推理成本。 |

## 先做哪三项

1. **P：在 `layer4_` 只试一处 BEVANet 式空间细节块。** [作者两个道路配置](p-evidence.md) 都在第二次 Pag 前用 EVA，实现可读。该块适合边缘断开、细杆或远处交通标志漏分。做法是先用可开关的 SDLSKA 空间块替换 P stage 4 的一个同宽块，保持 1/8 分辨率、Pag3/4、I、D 和主头；先比较原块与单块，确认有效后再隔离 CKS。文献中 EVA 同时影响两条分支，不能据此预报收益；1/8 上的分支激活和逐点投影都要测。RF-FEM 直接改 P 单项更有针对性，但作者代码与协议缺口较大，列为第二路线。
2. **D：在 `diff4` 加和后、`temp_d` 保存前加一处 GSCNN 式语义门控。** 将同级 I4 投影并对齐，只用它引导 D 的响应。该门控针对路面纹理、阴影或栅栏造成的伪边界；若 I 已漏掉细目标，语义门也可能误抑制。保持 `seghead_d`、`layer5_d`、Bag 接口及现有标签/损失，比较原 D 与单点门控，必要时加同预算的无门控对照。该位置同时影响辅助头和融合路径；只在 `layer5_d` 输出后加模块时，前向只改变融合输入。按 GSCNN 机制独立实现，不引入 Canny、ASPP 或 dual-task。
3. **P/I：确认存在位移后再试 `pag3` 的 2× DySample。** [PagFM](../../models/model_utils.py#L268) 目前对 I 的投影与值分别做双线性；改动时共享一套坐标，保持相似门与最终融合，先核对零偏移是否等价于原双线性。`pag3` 的 1/16→1/8 负担小于全面替换融合；放大不合整倍率时沿用原尺寸式插值。`pag4` 在 720 高 CamVid 上会遇到 23→90 的形状，不是整齐的 4×。若误差来自轮廓断续而非位移，优先回到 D 门控或局部 Bag，不为对齐硬上。

[本地落点与划分核验](notes.md) 显示 `seghead_d` 是训练时的单通道输出；Bag 用的是 `layer5_d` 的多通道特征。后续若获授权，先固定现有 I/边界设置、预训练来源、权重覆盖、输入和 seed，做 Cityscapes train→val 与 CamVid train→val 的单项对照；CamVid 原 YAML 的 `trainval` 已含 101 张 `val`，不能直接用来选方案。记录项目包括整体 mIoU、细类 IoU、去 void 的窄边界指标、同设备 batch 1 延迟和显存；方案固定后再用 test。论文增益不可相加，本地收益未知。

首轮未纳入：PILOT 另训新增类头，只回答 Cityscapes 15→19 的增量问题；LCFNets 虽有道路主干和配置，关键融合导入源码未在公开版本中定位；TSNet/HFINet 未获得可核作者实现；SparseRefine 需要稀疏算子和另一套修补流水线。详见 [D 证据](d-evidence.md) 与 [融合证据](fusion-evidence.md)。

> **交付记录（2026-11-11）：** Sol Lead 按原论文、作者源码和当前本地调用核对技术内容。首次通过 Super Agent 预约写作者时，监工已结算，预约因 `missing_support` 被拒。原工单保留 `blocked`，未验收。
>
> **修订（2026-11-11）：** 经用户明确授权，DeepSeek Writer 在原生会话完成润色提案。Sol Lead 对照原稿后采纳 7 处段落改写；候选表、数字、链接、归因、限制和优先序保持一致。
