# 本仓库落点和复核记录

> **Agent:** Sol Lead（gpt-6.1-sol / max）
> **Date:** 2026-11-11（Asia/Shanghai）
> **Ticket:** f3feb792cec5435daa96cf1ba825abc4

工作树 `main`，Git HEAD `4c158cf24ce432f0a8cb43364fae38d93cee0dc3`；调研开始时已有未提交的运行文件改动。以下接口以**现有工作树**为准，不假设已提交。未运行前向、训练、评估或外部代码，未检查 GPU 资源。

| 路径 | 当前代码 | 为什么要分开比较 |
| --- | --- | --- |
| P 空间特征 | [models/pidnet.py:190–206](../../models/pidnet.py#L190) 的 `layer3_`、`layer4_` 为 1/8；从 1/16、1/32 I 经压缩和两次 Pag。`temp_p` 来自 `pag3`，随后原地 ReLU 也整流这个别名；`layer5_` 走最终融合。 | 改 `layer4_` 不改变 P 辅助头在同次前向的直接取点；改 `pag3` 的输出则同时进入 P 辅助和后续 P。两者不混做一次消融。 |
| P/I 对齐 | [models/model_utils.py:290–311](../../models/model_utils.py#L290) 分别双线性对齐投影后的 `f_y(y)` 和值 `y`，`align_corners=False`。外层 logits [按配置](../../utils/utils.py#L43)，[默认值](../../configs/default.py#L32) 为 True。 | 单点对齐对照需要同一网格对两个 I 输入、保留内部和外部原有采样约定。精确 2× 是 `pag3` 的首选，形状不整除回退；不同时修改最终 logits 插值。 |
| D 共用特征 / 辅助边界 | [models/pidnet.py:204–225](../../models/pidnet.py#L204) 在 `layer4_d` 后加 `diff4` 的 I 投影，保存 `temp_d` 接 `seghead_d`（单通道）；其后一条支路送 `layer5_d`。`temp_d` 是别名，后续 `self.relu(x_d)` 为原地 ReLU，会同步整流该张量。 | 仅加模块到 `layer5_d` **输出**不会直接改该前向的辅助边界 logits。要在同一前向作用于训练边界头与最终融合，放 `diff4` 加和后、`temp_d` 保存前；若想只试融合门，则放 `layer5_d` 后。 |
| 最终融合 | [models/pidnet.py:214–226](../../models/pidnet.py#L214) 的 `dfm` 用 `layer5_d` 输出 `4*planes` 多通道 `sigmoid(d)`，不使用 `seghead_d` 的单通道图。[Light_Bag](../../models/model_utils.py#L314) 是 S/M 的两个 1×1 路径，[Bag](../../models/model_utils.py#L363) 是 L 的门控加权 + 3×3。 | 把单通道辅助边界概率直接换到 Bag 等于改接口与机制。记录 D 辅助 F 及最终语义边界指标，别用其中一个代表另一个。 |
| D 训练 | [数据目标](../../datasets/base_dataset.py#L134) 的 valid 邻接/膨胀、[边界损失](../../utils/criterion.py#L133) 的 mask/加权 BCE/可选 Dice，[BAS](../../utils/utils.py#L48) 的 `sigmoid()>0.8` 主语义损失与 void mask。开关 [当前默认](../../utils/ablation_config.py#L18) 仍为 legacy、mask off、Dice 0。 | 这些已是 D 改进能力；先固定选定的已验证配方。ABL 加在主多类 logits，和已有 D 标签、BCE/Dice、BAS 不同；可在后续验证里独立测试。 |

验证用文件名单已实际读出（每行首列图像路径去重）。Cityscapes `train/val/test` 为 2975/500/1525，`train∩val=0`；`trainval` 含全部 500 val。CamVid `train/val/test` 为 367/101/233，`train∩val=0`，`trainval∩val=101`。[CamVid S 基线 YAML](../../configs/camvid/pidnet_small_camvid.yaml#L11) 同时设 `TRAIN_SET=trainval.lst`、`TEST_SET=val.lst`，因此未来调参必须用实验副本/覆盖的 `train.lst → val.lst`；测试集不参与超参、阈值或 checkpoint 选择。Cityscapes 同样用 train→val；仅固定后方案另作 test。Cityscapes/CamVid 都固定预训练来源及覆盖率、split、边界与 I 开关、输入尺度、seed 和计时协议。

建议记录整体 mIoU、类 IoU（Cityscapes 的 pole、traffic light、traffic sign、rider 等；CamVid 使用其实际类表）、固定容差的窄边带指标，以及同设备原分辨率 batch 1 延迟和峰值显存。边带剔除 void，固定宽度后再比较；把漏检、轮廓模糊、P/I 位移和类别混淆分开。若首轮差距小，对有效候选重复 seed；只做单项后再决定是否组合。

数值核对：用作者原表做十进制差（如 BEVANet BGAF 79.27−78.86=0.41、DySample 43.58−41.68=1.90、FreqFusion 80.1−78.5=1.6、Agri 单 BWR 91.79−88.84=2.95）；论文 FPS 不作本地预估。详见 [P 证据](p-evidence.md)、[D 证据](d-evidence.md) 和 [融合证据](fusion-evidence.md)。
