# P 分支与 Pag：证据核对

> **Agent:** Sol Worker（gpt-6.1-sol / high）
> **Date:** 2026-11-11（Asia/Shanghai）
> **Scope:** PIDNet 的 P 结构、P/I 中段融合及 P 末端增强。只读论文和源码，无本地模型运行。

## 本仓库的接口与验证口径

Xu、Xiong、Bhattacharyya，[PIDNet，CVPR 2023，§3.1–3.2](https://arxiv.org/html/2206.02066v3#S3.SS2)，P 是高分辨率细节，I 是低分辨率语义，D 是边界。当前工作树 [PIDNet 构造](../../models/pidnet.py#L48)、[forward](../../models/pidnet.py#L182) 中，`x_` 为 P，`layer3_` 和 `layer4_` 保持 1/8、`2*planes` 通道（S=64，M/L=128）；`pag3`/`pag4` 分别取 1/16、1/32 I 特征，经 `compression3/4` 映射到同通道并在 [PagFM](../../models/model_utils.py#L268) 中按 P 尺寸双线性插值。默认 gate 是投影逐点乘积后沿通道求和再 sigmoid，P/I 插值混合共用一个空间门图。P 辅助头用第一次 Pag 后的 `temp_p`；`layer5_` 输出 `4*planes`（S=128，M/L=256）送 `dfm(x_, x, x_d)`。[原论文图 8](https://arxiv.org/html/2206.02066v3#S4.SS3) 的第二个门图在示例中对小目标更暗，符合保留 P 的设计；是可视化证据，无单类定量保证。后续应先选一个 P 落点；固定现有 I context/atrous、D 边界监督及最终 Bag，避免同一轮用多个变化解释收益。

成本尺度：Cityscapes 原图 1024×2048 的 P 图为 128×256（32,768 点），CamVid 原图 720×960 为 90×120（10,800 点）。全幅 Cityscapes 上一张 C=64/128/256 的 FP16 特征为 4/8/16 MiB；一次同宽普通 3×3 为约 1.21/4.83/19.33 GMAC（单样本、只计乘加），可见 P 端少量参数也可能带来可见推理成本。若后续获授权，固定训练划分、预训练/seed、原图单尺度无 flip 及速度测试条件；同时看全局 mIoU、pole/traffic light/traffic sign/rider 等类的 IoU、边界指标和峰值显存，记录误差是否来自细节丢失还是语义误注入。不要拿不同协议的文献 FPS 作本地预期。

## 直接改版：RSM-Optimizer / RF-FEM

Xiaohong Zhang、Wenwen Zong、Yaning Jiang，2025，[RSM-Optimizer: Branch Optimization for Dual- or Multi-Branch Semantic Segmentation Networks](https://www.mdpi.com/2079-9292/14/6/1109)，Electronics 14(6):1109。§3.1–3.2、图 1–2 和式 1–3 给 P 端结构：1×1 压宽后接三级串联 3×3，取压宽结果及各级中间特征，分别接 dilation 1/3/5/7 的 3×3；拼接压缩与 1×1 shortcut 相加。它在 P 内增加多尺度空间感受野，和现有 I 低分辨率上下文不同。§3.3 的 SUFM 在最终融合前逐级上采样 I，包含 CARAFE/WFB；移植 RF-FEM 可保持原 Pag、SPP、Bag。原文用图定位 P 分支，未给可验证的 Python stage 名、内部宽度、规范化/激活及与原 residual block 的替换关系，以下本地落点属于建议。

[表 1](https://www.mdpi.com/2079-9292/14/6/1109#electronics-14-01109-t001) 的 Cityscapes PIDNet-S 同文消融（单位：mIoU %、FPS、M、GFLOPs）：

| 同文设置 | mIoU | FPS | 参数 | GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| S 基线 | 78.3 | 125 | 7.62 | 47.6 |
| RF-FEM | 78.8 | 120 | 7.92 | 57.2 |
| PMPM | 79.1 | 103 | 8.41 | 47.9 |
| RF-FEM + PMPM | 79.4 | 98 | 8.71 | 57.7 |
| RF-FEM + PMPM + SUFM | 79.8 | 94 | 9.92 | 69.3 |

RF-FEM 单项 +0.5 pp、−5 FPS、+0.30M、+9.6 GFLOPs；已有 PMPM 时加 RF-FEM 是 +0.3 pp。SUFM 没有独立行，+0.4 pp 是在前两项上叠加。表 2 的 S/S* 是 2048×1024，正文称最高但表 2 PIDNet-M 为 80.1，高于 79.8。表 3 的 CamVid S 80.1/153 → S* 82.8/150（720×960）是全套 +2.7 pp，不能归给 P；表 4 的 pole 66.1→69.1、traffic light 72.6→73.9 也只属全套。§4.1 列 Cityscapes 19 类 fine、CamVid 11 类，训练/验证/测试 2975/500/1525 和 367/101/233；Python 3.8.19、PyTorch 2.5.0、5 张 RTX 4090 服务器、SGD、batch 6、500/200 epoch。表 1–4 没把每个数字绑定 val/test、初始化、测试尺度/flip 或精确测时设备与批量。原始 PIDNet [表 5、6](https://arxiv.org/html/2206.02066v3#S4.SS4) 的 S 数值属于原作者协议，不用于替这篇确定 split。

代码：正文 Data Availability 只链接 Cityscapes 数据，未给作者仓库。以题名、DOI、作者、模块名搜索网页、GitHub 和 Gitee 索引，并用 GitHub 文件搜索 `RF_FEM RFFB`，未定位能归属作者的实现或配置。论文 CC BY 4.0 不等于源码授权；作者软件许可证、依赖锁定与复现实验入口未核。结构由标准卷积组成，可按图式独立实现 RF-FEM 近似版；CARAFE 属于另外的 SUFM，首轮不用它。

落点：先取 `models/pidnet.py:203` 后、`:206` 第二个 Pag 前的单个 1/8 P 特征，在 `2*planes→2*planes` 的窄瓶颈内实现，保持第一次 Pag 的辅助监督及最终 D/I 融合。与已选 I context 同设比较单模块，若有必要再分「多尺度卷积」与「更深/更多参数」的对照。小型交通物及细长结构漏分是适用假设；大 dilation 在稀疏细节上有采样空洞风险，四路中间激活及拼接会增加带宽。作者未给可加载 RF-FEM 权重，不能把独立实现当作完全复现。

## 直接改版、主域为唐卡：ECA-Pag / LGFM

Jiao Wu、Tiejun Wang、Xiaoyan Hu、LingMei Tao、Tianjiao Duan、Yanjiao Wei，2026，[Research on the semantic segmentation of Thangka images via an improved PIDNet](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0332164)（[出版社 PDF](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0332164&type=printable)），§2.2–2.3/图 3–4。论文在 P 两次 Pag 加 ECA 及残差，P 末端加 LGFM；同一整网另在 D 加 PConv3。下述实现读自论文链接的 [作者库](https://github.com/JikD-591/thangka-seg)，提交 `cbcb6de56e0e777c17e1b649504d12006900c024`，未运行。

- **ECA-Pag 默认并未做逐 P 通道重权重。** 作者的 [PagFM/ECAAttention](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/mmseg/models/backbones/newpidnet.py#L142-L237) 和 [两处构造](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/mmseg/models/backbones/newpidnet.py#L344-L358) 显示 `with_channel=False`。设原始通道求和相似度为 `s`，ECA 在 `B×1×H×W` 的 `s` 上做空间平均、长度 1 上的带 padding `Conv1d(1,1,3)`、sigmoid，得到每张图一个缩放系数 `g`；门为 `sigmoid(g*s)`，两次混合结果还各加 `x_p`。真正的多通道 ECA 要启用额外 projection 路径或前移到求和前，两者均属于新变体。默认 `0<g<1` 会把门推向 0.5，未必使 P 更可靠；残差使 P 权重从 `1−sigma` 变成 `2−sigma`，也改变下一层的幅度。
- **LGFM 的全局注意力矩阵按通道，非按像素配对。** [局部 CoT、区域卷积和全局实现](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/mmseg/models/backbones/newpidnet.py#L26-L107) 可见：全局路径 `1×1` qkv 投影重排为 `B×heads×C_head×HW`，沿 `HW` 归一化，注意力矩阵 `B×heads×C_head×C_head`；局部 CoT 以 DW 3×3/key 和 1×1/value 拼接，注意权重按最后一维 softmax；另有 DW 3×3 + 1×1 区域支路，三项直接相加。定义的 `project_out` 未在 `forward` 调用；论文式 5 的 Dconv 不在发布代码的全局 qkv 路径。实际 [forward](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/mmseg/models/backbones/newpidnet.py#L510-L569) 在 P 最后一层之后、`dfm` 之前调用 LGFM。D 中 `Partial_conv3` [实现](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/mmseg/models/backbones/newpidnet.py#L112-L140) 是只卷积部分通道，没有论文所述有效像素空间掩码。

同域数字采用 [表 5，PDF 第 14 页](https://doi.org/10.1371/journal.pone.0332164.t005) 的 5 seed 平均；唐卡 PIDNet-M 基线 70.20±0.17 mIoU、35.97±0.13 mB-Fscore、104.63 FPS；ECA-Pag 单项 70.76、35.94、101.50（+0.56/−0.03 pp）；PConv3 单项 72.00、36.46、117.12；LGFM 单项 74.14、36.41、85.98（+3.94/+0.44 pp）；全套 73.33±0.12、40.06±0.17、109.06（+3.13/+4.09 pp）。LGFM 单项 mIoU 更高，组合边界分更高。§2.5–2.8 列 512×512、Adam、batch 8、45k 上限；783 原图扩至 4,698、描述为 3,776/922，训练内留 755，未证实按原图分组隔离；唐卡预训练与测试增强未核。摘要值与表 2/5 均值略异，采用表值。

道路只见 [Cityscapes 表 8，PDF 第 19 页](https://doi.org/10.1371/journal.pone.0332164.t008)：PIDNet 74.38 → Ours 75.40（+1.02 pp），都是 RTX 3090、1024×1024 crop、80k；512×512 是同表 Swin-B 行。该表没有 P 单项、PIDNet 具体变体、val/test、预训练、TTA、道路计时；不能把唐卡单项收益外推。打开的作者 [Cityscapes S](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/configs/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes.py#L24-L103)、[M](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/configs/pidnet/pidnet-m_2xb6-120k_1024x1024-cityscapes.py)、[L](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/configs/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes.py) 为普通 `PIDNet`、120k、1024 crop 和 ImageNet 初始化（M/L 继承 S），无法当表 8 复现配置。源码 [有注册入口](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/mmseg/models/backbones/__init__.py#L17)，但本轮没有定位到对应 80k 改版 YAML/Python 配置。论文链接 [Zenodo](https://zenodo.org/records/19498530) 含唐卡数据/模型，未取文件。

公开材料和成本：作者 [README](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/README.md#L9-L29) 写 Python 3.8+/PyTorch 1.10+；[requirements.txt](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/requirements.txt) 列 torch、torchvision、mmcv-full、openmim 等，无版本锁。源码另导入 MMEngine、MMSeg、einops；版本组合未验证。固定提交根 `LICENSE` 返回 404，未核到新改动的独立授权，不能用论文 CC BY 或源码中的 OpenMMLab 标注推定。[作者改版构造](https://github.com/JikD-591/thangka-seg/blob/cbcb6de56e0e777c17e1b649504d12006900c024/mmseg/models/backbones/newpidnet.py#L405-L412) 写死 LGFM 256 通道及 D 的 64/128 通道；若原样套公开 S 配置，LGFM 输入 128 会不匹配；若套公开 L，D 第一层产出 128，与 `block64` 固定拆分也不匹配。此处是静态尺寸推导，未运行。

可从 `models/model_utils.py:306` 单独试第二个 Pag 的作者标量门和 P residual，分别报告 residual-only 与 `g`+residual，便于查语义误注入，且几乎不增加参数。LGFM 仅建议 P 末端 `models/pidnet.py:214` 后作二线；若通道随本地变体设置，Cityscapes 全幅 qkv 单张张量在 S/M-L 分别约 24/48 MiB FP16（不含梯度、中间量）。4 heads 下仅卷积与注意矩阵乘约 4.10/16.26 GMAC，未计归一化、softmax 与访存；不等同已测延迟。

## 新的有作者实现的街景近邻：BEVANet / EVA

Ping-Mao Huang、I-Tien Chao、Ping-Chia Huang、Jia-Wei Liao、Yung-Yu Chuang，2025，[BEVANet: Bilateral Efficient Visual Attention Network for Real-Time Semantic Segmentation（ICIP）](https://ieeexplore.ieee.org/document/11084676)，[作者全文 v1](https://arxiv.org/html/2508.07300v1)。作者代码从 PIDNet 派生，但 BEVANet 主体为双分支；其 [模型](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/models/BEVANet.py#L406-L450) 在 P 对应的细节特征末端才分出 boundary Bottleneck，由 BGAF 融合，没有 PIDNet 的独立多级 D。因此是 P 近邻迁移，不能用整网差称 PIDNet 同设增益。

作者 [GitHub](https://github.com/maomao0819/BEVANet) 提交 `52a504960a08d943cb8dd97c0d540ee2b9e1f46f`，已打开 [Cityscapes 配置](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/configs/cityscapes/BEVANet.yaml)、[CamVid 配置](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/configs/camvid/BEVANet_S.yaml) 及 [高分辨率块构建](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/models/BEVANet.py#L92-L130) / [调用](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/models/BEVANet.py#L165-L202)。两道路配置是 `DETAIL_KERNEL_SIZE: [0,23]`：第 3 阶段 1/8 细节沿用普通块和第一个 Pag，第 4 阶段是一个 BasicBlock → `EVAN` → LayerNorm → 第二个 Pag。其 [SDLSKA](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/models/LSKA.py#L123-L152) 对 `k=23` 先做 DW5×5，再以 dilation 3 的 DW1×7/7×1 并联从小核输出取横/纵特征（有效跨度 23），经 [CKS](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/models/FS.py#L11-L67) 用通道与空间 sigmoid 加权，再用 pointwise 投影与输入逐点相乘。完整 [EVAN](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/models/EVAN.py#L49-L122) 还有残差、归一化、1×1 投影和 4 倍扩宽 MLP。作者 [LICENSE](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/LICENSE) 为 MIT；[requirements.txt](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/requirements.txt) 为 torch、opencv-python、yacs、tqdm、tensorboardX，无版本锁，读取的 EVA 路径用标准 PyTorch 算子。源码分别注明来自 LSKA、RTFormer、LSKNet 和 PIDNet 的修改，若直接移代码要保留归属并核各上游许可；未跑安装或训练。

| 作者全文表 | 路面实测及口径 | 能说明什么 |
| --- | --- | --- |
| 表 1 | ImageNet 预训，Cityscapes **val**：81.0 mIoU、32.8 FPS、238.2 GFLOPs、58.6M，2048×1024，RTX 3090、batch 1。 | 完整 BEVANet，横向 PIDNet 多为引用，不能据此归因 EVA。 |
| 表 2 | BEVANet-S：CamVid 83.1 mIoU、79.4 FPS、20.1 GFLOPs。表内未标 split；作者 [README](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/README.md) 和对应配置明确为 val，打开的 [val.lst](https://github.com/maomao0819/BEVANet/blob/52a504960a08d943cb8dd97c0d540ee2b9e1f46f/data/list/camvid/val.lst) 有 101 个非空条目。 | 表 2 位于预训比较段，CamVid 是否还经 Cityscapes 微调未核；所引 PIDNet 原论文表 5 是 CamVid test，故不能宣称同协议 +1.1 pp。 |
| 表 3 | 无 ImageNet：Cityscapes 79.27 / 32.9 FPS，表未逐行给出 split。 | 完整 BEVANet，不能作 P 单项。 |
| 表 4 | Cityscapes 上下文、无 ImageNet；BA+普通卷积+Bag 77.77/44.25 → SDLSKA+加法 78.60/37.79 → SDLSKA+CKS 78.86/37.29。 | 固定 BA/Bag 时 +0.83 pp，再 +0.26 pp；未见只改 P 或仅换第二处的表项。CKS 相对 LSKNet 78.72/37.46 实为 +0.14 pp、−0.17 FPS，正文误把对加法的差写给 LSKNet。若同时换 BGAF，78.86→79.27（+0.41 pp、−4.44 FPS），不是 EVA 的增量。 |

两发布配置均 `TEST.FLIP_TEST=false`、`TEST.MULTI_SCALE=false`，但表 3/4 未单列 split/TTA，不能由配置追认；两 YAML 的默认 `PRETRAINED` 都为空，作者 README 给参数覆盖；Cityscapes YAML 500 epoch 对正文 484、CamVid YAML 100 对正文 200，说明发布配置不能被当成所有表项的原运行配置。作者 PDF 原表在第 4–5 页；正文和原表冲突以上表为准。

落点：按源码改当前 `models/pidnet.py:203–206`，先在 `layer4_` 与 `pag4` 间试单个窄版 SDLSKA；若用作者等价完整 EVA，将第二组 P 残差块的一块换为 EVAN，而非把整套 P/I/D 一起替换。固定 I atrous/PMPM、Pag 门、D 及 Bag，再比较条带+加法与条带+CKS。假设对断裂细杆、标志及窄轮廓有用；条带稀疏取样可能受方向及边界 padding 影响。标准 depthwise 让 23 的空间跨度不等于密集 23×23，但分支激活、逐点投影、CKS、扩宽 MLP 和 LayerNorm 都在 1/8 发生，必须实测延迟/显存；不要套作者整个网络的 FPS。

## 检索记录与取舍

| 实际查询或打开 | 取舍 |
| --- | --- |
| `PIDNet "RSM-Optimizer" "RF-FEM" GitHub code semantic segmentation`、`"RSM-Optimizer" "Wenwen Zong" GitHub Gitee "RF-FEM"`、`"RF-FEM" "SUFM" PIDNet code OR GitHub OR Gitee 10.3390/electronics14061109`、`site:github.com "10.3390/electronics14061109" OR "Receptive Field-Driven Feature Enhancement"`、`site:gitee.com "RSM-Optimizer" PIDNet OR "RF-FEM"`、GitHub 文件 `RF_FEM RFFB`；打开 §3–4/表 1–4、Data Availability、图 1/2 资源与评审页。 | 未定位作者软件；GitHub 文件查询同名结果为无关文本。MDPI 直链 PDF 超服务大小上限，图资源浏览器连接关闭；不声称已从图片辨认具体 stage。 |
| 论文 §2.8 的 `JikD-591/thangka-seg`；固定提交核 `newpidnet.py`、注册文件、S/M/L 配置、数据基配置和 README；GitHub 文件 `newPIDNet`。 | 模块实现可读，公开道路配置是旧基线，未定位表 8 的改版配置或源码许可。出版社 PDF 第 14/19 页发起截图请求，工具只返回图像占位；列位另外按 PDF 文本与正文对核。 |
| `Cityscapes semantic segmentation "detail aggregation" high resolution author github detail branch DANet CABiNet 2025`、`Cityscapes real time segmentation detail branch spatial detail module ablation GitHub RTFormer DABNet Detail Aggregate high-resolution`、`BEVANet Real-Time Semantic Segmentation cityscapes detail branch author 2025 BGA arxiv paper`、`"BEVANet" "github" Cityscapes "Ablation"`、GitHub 文件 `class SDLSKA` / `class CKS`。 | 确认 BEVANet 在旧调查文件未出现，论文、作者块和两街景配置能互证。FBSNet 有浅层细节支路，[作者 PDF](https://guangweigao.github.io/paper/TMM-FBSNet.pdf) 和 [代码入口](https://github.com/IVIPLab/FBSNet) 可见；DABNet 有 [作者项目入口](https://github.com/isyangshu/DABNet)。两者未继续逐文件核，不计为同等级候选。 |
| `"Context-Guided Detail Fusion" CGDFNet github code source`、`"TBMSNet" segmentation github code triple branch multi-scale 2026 author`，查文章 Data/Code Availability 和分支说明。 | [CGDFNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC12996344/) 的 CGDM 是贴近 P 的新线索，正文只连数据，未定位作者代码；[TBMSNet](https://www.nature.com/articles/s41598-026-48759-x) 保留 Pag，重点在 I 的 DWR/MSAPP，作者声明代码可请求，旧报告已核。二者不替代本次新增有代码的 P 候选。 |

建议顺序：有明显细物体漏分先做 RF-FEM 单位置并以有 MIT 代码的 SDLSKA 作为可复查替代；若错误集中在语义特征压过边缘细节，则先做便宜的 Pag 残差/标量门对照。LGFM 的道路证据无法拆出，且 1/8 激活成本高，排在两者后。以上为结构建议与文献数字，未改模型、未训练、未评估、未执行远端代码。
