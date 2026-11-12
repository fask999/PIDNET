# PIDNet 融合证据与迁移边界

> **Agent:** Sol Lead（gpt-6.1-sol / max）
> **Date:** 2026-11-11（Asia/Shanghai）

以下数字是作者结果。已打开的实现和配置按所读版本链接；未运行代码、未下载权重。表中增量是百分点。

## P/I 对齐：DySample 与 FreqFusion

**DySample，Wenze Liu、Hao Lu、Hongtao Fu、Zhiguo Cao，ICCV 2023，*Learning to Upsample by Learning to Sample***。[作者原文 §3、§4.1、表 4](https://arxiv.org/pdf/2308.15085#page=7)，[作者仓库](https://github.com/tiny-smart/dysample)，已打开 [算子](https://github.com/tiny-smart/dysample/blob/81a1de5caa95d55a0f5488425fa53ec7ef47f8f0/dysample.py#L20-L82) 和 [MIT LICENSE](https://github.com/tiny-smart/dysample/blob/81a1de5caa95d55a0f5488425fa53ec7ef47f8f0/LICENSE)。

- 分组学习坐标偏移，用 `pixel_shuffle` 排列细网格，`grid_sample` 双线性取样，`align_corners=False`、边界复制；不需要高分辨率引导输入。公开模块依赖 PyTorch，不依赖自定义 CUDA 扩展；整数放大倍数和组数/通道有约束。作者小仓没有本次可核的道路训练配置。
- 表 4 是 ADE20K 上的 SegFormer-B1 对照，双线性 41.68 mIoU / 27.80 bIoU，DySample-S+ 43.58 / 29.93（+1.90 / +2.13）。表题未逐行说明 split、权重或 TTA；不借别的论文把这些口径补齐。原文讨论认为该结果主要改善内部区域；依赖高分辨率引导的 FADE/SAPA bIoU 更高。图 1 的 1.6→6.2 ms 是单个 256×120×120 特征的 2× 运算对比，不能估算本仓延迟。
- 接入：[PagFM](../../models/model_utils.py#L268) 的 `pag3`：I 路先做原有 `compression3`；共享一套偏移网格给 `f_y(y)` 和原 `y`，保留 `f_x`、相似度和 Bag。分组适配先用 `groups=1` 隔离对齐变量；数值先对比零偏移与当前尺寸式 bilinear。当前 PIDNet 的全局 Kaiming 初始化会覆盖普通 `Conv2d` 的特殊初始化，未来实现需明确处理。CamVid 720px 输入的 P 为 90 高，I stage 3 为 45 高，stage 4 为 23 高；直接以固定倍率替 `pag4` 会得到 92 高。保留非精确倍率时原尺寸式双线性路径，另验 AMP/部署 `grid_sample`。

**FreqFusion，Linwei Chen、Ying Fu、Lin Gu、Chenggang Yan、Tatsuya Harada、Gao Huang，TPAMI 2024，*Frequency-Aware Feature Fusion for Dense Image Prediction***。[发表 DOI](https://doi.org/10.1109/TPAMI.2024.3449959)，[作者全文 §V-A、表 II/IV/VIII](https://arxiv.org/html/2408.12879v1#S5)，[作者仓库](https://github.com/Linwei-Chen/FreqFusion)，已打开 [FreqFusion.py](https://github.com/Linwei-Chen/FreqFusion/blob/3fb0c70637a3c194fb74294d3ce4681958b26241/FreqFusion.py#L112-L277)。

- 低通平滑深层特征，高通残差增强浅层特征，再可选局部相似度偏移。返回 mask、增强后的高分辨率输入及对齐后的低分辨率输入，不能直接替换单输入 `interpolate`。`feature_resample=False` 是这份算子的默认值；复现完整机制时要显式记录开关。默认 semi-conv 路径的 CARAFE 调用与偏移器围绕 2× 设计。
- 表 IV 是 Cityscapes **val 单尺度**，SegFormer-B1 78.5→80.1（+1.6），243.7→271.8 GFLOPs（1024×2048，+28.1）；UPerNet-R50 78.8→79.8（+1.0）。§V-A 说沿用原模型配方（SegFormer 和 SegNeXt 为 Cityscapes 1024×1024 crop、160k）；未把每行预训练权重及是否翻转与特定表行/配置绑定。表 VIII 另在 SegNeXt-T、RTX 3090 同分辨率测 26.5→23.0 FPS。表 II SegFormer-B1 的 bIoU 27.8→32.8 是 **ADE20K val**，不能冒作 Cityscapes 边界指标。多处融合一起换，不能视为单个 Pag3 的收益。
- [依赖入口与回退](https://github.com/Linwei-Chen/FreqFusion/blob/3fb0c70637a3c194fb74294d3ce4681958b26241/FreqFusion.py#L3-L56) 为 PyTorch/NumPy，优先 MMCV CARAFE；纯 Torch 例程使用 `unfold`、显式整倍数及单组形状，不能假定不同后端的速度、显存和行为相同。主仓根 `LICENSE`、`LICENSE.md` 访问返回 404，README 未找到授权声明；主算子复用许可未核，不能用子项目许可证推定覆盖。未找到 PIDNet 配置。
- 先做同一处 `pag3` 的廉价坐标对照。如果能验证模糊边界还伴随区域内纹理不一致，再研究 P 作高分辨率端、压缩 I 作低分辨率端的 FreqFusion；单独比较「仅对齐 I，P 保持原值」与「一起启用 P 高通」，确保 key/value 同网格。没有证据表明需要同时换 `pag4`、D 注入、最终 logits 或 I 的上下文模块。

## 最终融合：直接 PIDNet 改版和道路近邻

**Tan / Jin，2025，*A Semantic Segmentation Method for Road Sensing Images Based on an Improved PIDNet Model***。[作者正文 §4.3、§5.2–5.4、表 1–3](https://www.mdpi.com/2079-9292/14/5/871)。在 D 门控的加权融合基础上增加 P/I 特征交互；完整模型另有 HRNet 式共享提取和 DenseASPP，后两项本轮不混入。

- 原刊表 3 没有 cross-attention 单项。按表中成对行看，在已有 DenseASPP 时 79.0→79.5（+0.5），在已有 HRNet + DenseASPP 时 80.5→81.1（+0.6）。在 HRNet 上 79.1→79.9（+0.8），但该行叙述写 79.7，采用表值并保留冲突。§5.2 写 ImageNet 初始化及 Cityscapes 620 epoch；表 3 没标 val/test 或 TTA，78.6 原版行不能证明同设重训。表 1 单独列 val 78.8→81.0、test 78.6→81.1 的**全套**横比；表 2 CamVid 80.1→82.6 也是全套，split/TTA 未明。表 4 只有 16 项，不能作为 19 类成绩。
- 已查论文 Data Availability 和题名/DOI/作者/`cross attention Bag` + GitHub/Gitee，未定位完整作者模型、配置或代码许可。正文公式说明思路，但张量布局和训练细节不足以复刻。只作为 [dfm](../../models/pidnet.py#L221) 的融合假设；要做可选择局部/低秩候选，单独对照原动态 `sigmoid(d)`，不要把基线描述成固定权重。全图点对点注意力在 Cityscapes 1024×2048 的 1/8 网格有 32,768 token，每个 head 若显式构建矩阵约 10.74 亿个分数。

**Agri-PIDNet-S，Yanlin Xin 等，PLOS One 2026，*Agri-PIDNet-S: Lightweight semantic segmentation for field-structure and obstacle perception in seedling-stage cotton fields***。[原文](https://doi.org/10.1371/journal.pone.0361992)，[表 4 全因子](https://doi.org/10.1371/journal.pone.0361992.t004)、[表 5 内部件](https://doi.org/10.1371/journal.pone.0361992.t005)。已打开作者 [BWRBag](https://github.com/xyl-design/xinyanlin/blob/1b1938e7611189d860d935187d93a27af76caa16/agri_pidnet/model.py#L486-L552)、[选择处](https://github.com/xyl-design/xinyanlin/blob/1b1938e7611189d860d935187d93a27af76caa16/agri_pidnet/model.py#L677-L684)、[配置](https://github.com/xyl-design/xinyanlin/blob/1b1938e7611189d860d935187d93a27af76caa16/configs/agri_pidnet_s.py)、[实验矩阵](https://github.com/xyl-design/xinyanlin/blob/1b1938e7611189d860d935187d93a27af76caa16/configs/experiment_matrix.yaml)、[训练入口](https://github.com/xyl-design/xinyanlin/blob/1b1938e7611189d860d935187d93a27af76caa16/tools/train.py#L68-L69) 和 [MIT 许可](https://github.com/xyl-design/xinyanlin/blob/1b1938e7611189d860d935187d93a27af76caa16/LICENSE)。项目栈 PyTorch 2.1、MMEngine 0.10.3、MMCV 2.1、MMSeg 1.2.2；窗口核心是 PyTorch。

- BWRBag 以 D 对 P/I 加权，先过 BN/ReLU/3×3，再做 4×4 非移位窗口、两头自注意力和 FFN 的残差细化；边缘尺寸用 replicate pad 后裁回。关掉 `use_bwr_bag` 时实际切回 LightBag，故表 4 的 BWR-only 同时换了基础融合与窗口。表 5 只关窗口的对照更干净：90.37→91.79 mIoU（+1.42）。
- **四类棉田测试集**，495/107/107 固定划分。表 4 同一协议 seed 0、终步 20k、同初始化协议，原 PIDNet-S 88.84→只开 BWR 91.79（+2.95）→全套 92.37；全套另做三 seed，但单项只跑一次。具体初始化权重未核，发布配置 `load_from=None`、CLI 可覆盖；评估公开入口是一次 whole-image 前向，原表未另列 TTA。不能把数值外推道路。
- 道路若主要是边界断续、细长类局部错分，可保留原 `dfm`，只在其输出到 `final_layer` 间挂独立窗口残差做对照；这是借用机制，不是原 BWR 全量替换。对比窗口边界伪影和显存，先独立于 I、D 结构和损失；不把其边界 BCE 换入本仓作为同一变量。

**BEVANet，Ping-Mao Huang 等，ICIP 2025，*Bilateral Efficient Visual Attention Network for Real-Time Semantic Segmentation***。[作者论文表 4](https://arxiv.org/pdf/2508.07300v1#page=5)，[作者 BGAF、Bag、Light_Bag 同文件](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/models/FB.py#L8-L71)。相同 BA + SDLSKA + CKS、无预训练的 Cityscapes 消融，Bag 78.86 / 37.29 FPS、Light Bag 78.39 / 38.48、BGAF 79.27 / 32.85；相对 Bag +0.41 / −4.44 FPS。表 4 没单独写 val/test 或 TTA；正文同一节是 Cityscapes，道路 val 横表和发布配置另看 [P 证据](p-evidence.md)。代码把 P/I 各做一层 3×3，再把有门控的加权和加到无门控 `p+i`，最后一层 3×3；三个全通道 3×3 都在高分辨率处。相同接口可在 `dfm` 做直接切换，S/M 基线是两路 1×1 Light_Bag，L 是 3×3 Bag，要按实际变体比较开销。BGAF 只吃末端多通道 D，不改辅助头。它是 BEVANet 的独立融合对照，不是 PIDNet 的直接增益。

**任凤雷等，2026，《基于三分支网络的实时图像语义分割》**。[作者期刊](https://ope.lightpublishing.cn/zh/article/doi/10.37188/OPE.20263401.0167/)，[期刊 PDF §3.3、表 1/3/4](https://ope.lightpublishing.cn/rc-pub/front/front-article/download/146561864/lowqualitypdf/Segmentaci%C3%B3n%20sem%C3%A1ntica%20de%20im%C3%A1genes%20en%20tiempo%20real%20basada%20en%20red%20de%20tres%20ramas.pdf#page=7)。P/D 末端空间 GAP 后沿通道序列做五路 1D 卷积（核 3、dilation 1/3/5/7/9），求和 sigmoid 再乘原特征；不是在图像上空洞卷积。表 3 作者三分支网络 Cityscapes 78.7（无注意力）→79.2（两末端 MCIAM，+0.5），无 P-only/D-only；原作者取消 Pag、保留 Bag，消融 split、初始化与 TTA 未明确。已读表 1 PyTorch 风格实现，按 DOI/题名/作者/MCIAM 搜未定位完整仓、训练配置或代码许可。若借鉴，先在一个末端单独加，P 用 `layer5_` 输出、D 用 `layer5_d` 输出（只影响最终融合），另一端和 Pag 保持原状。

## 暂不进首轮

- **LCFNets（Lu Yang、Yiwen Bai、Fenglei Ren、Chongke Bi、Ronghui Zhang，IEEE T-IV 2024）**：[发表记录](https://doi.org/10.1109/TIV.2024.3363830)，[作者仓库](https://github.com/yiwen-bai/LCFNets)，已打开 [主干](https://github.com/yiwen-bai/LCFNets/blob/2ea754a7a0e010b5fe81bf33865354acb27e48f6/models/lcfnet.py#L162-L228)、[Cityscapes 配置](https://github.com/yiwen-bai/LCFNets/blob/2ea754a7a0e010b5fe81bf33865354acb27e48f6/configs/cityscapes/lcfnet_slim_cityscapes.yaml)、[CamVid 配置](https://github.com/yiwen-bai/LCFNets/blob/2ea754a7a0e010b5fe81bf33865354acb27e48f6/configs/camvid/lcfnet_slim_camvid.yaml) 和 [依赖文件](https://github.com/yiwen-bai/LCFNets/blob/2ea754a7a0e010b5fe81bf33865354acb27e48f6/requirements.txt)。作者代码从 PIDNet/HRNet 派生，但重新布局为 1/4 detail、semantic、compensation；`x_d` 指 detail 而非 PIDNet D。`L_DGF`/`L_SGF`/`TGF` 从 `models/model_utils.py` 导入；指定分支的 API 和 raw 都返回 404，发布提交 [c997094](https://github.com/yiwen-bai/LCFNets/commit/c99709424e88a354a6fda40847a39bb1d16051f1) 只有该模块的缓存字节码条目，未查到公开源码或项目许可，未运行字节码。IEEE 正文/原表未能打开，作者 README 的整网横比只作线索。无法据此给 TGF 可复现落点。
- **TSNet / HFINet**：[TSNet 出版页](https://www.sciencedirect.com/science/article/pii/S1051200426003969) 的 USM/LUSM、BGFM 与 [HFINet 出版页](https://www.sciencedirect.com/science/article/abs/pii/S0923596525001213) 的边界融合有结构相关性；定向搜索没有找到可核的作者实现，本轮可取得的材料不足以确认对应单项配置和消融，不据摘要分数给可执行推荐。

实际检索：`PIDNet Bag Cityscapes cross attention fusion github`、`PIDNet PagFM Cityscapes boundary detail github`、`PIDNet boundary-aware fusion Cityscapes CamVid RGB GitHub`、`TSNet BGFM LUSM github`、`LCFNets 3363830 ablation table`、`LCFNet L-DGF TGF github model_utils`、`Tan Jin 10.3390/electronics14050871 cross attention Bag code`、`MCIAM EPPM GitHub Gitee`、`Frequency-Aware Feature Fusion Cityscapes`。从出版社、作者 arXiv 和仓库追到原表、固定提交、代码与配置。网页截图接口对上述 PDF 仅返回占位，表值以可读原 PDF 文本和出版社 HTML 表核对，不把截图占位当成视觉证据。
