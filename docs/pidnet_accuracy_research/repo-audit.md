# PIDNet 本地源码与准确率协议审计

范围：`/home/robot/catkin_ws/PIDNET`，源码基线 `4c158cf`。只读核对模型、损失、训练入口、验证/测试入口、数据管道、8 份 YAML 和 8 份列表；用合成张量做了小型 CPU 检查。没有训练、修改运行代码、读取实际图像/权重或重测 mIoU。下文的“已确认”限于源码、列表或明确列出的 CPU 现象；“风险”注明触发条件；“设计”描述当前行为，改动需要对照实验。

先固定基线协议：Cityscapes 的独立 val 用 `train` 配方和 `val.lst`；`trainval` 配方的训练中 val 是样本内读数。CamVid 自带配方也使用 trainval 和样本内 val，报告测试成绩用独立 `test.lst`。评估须核实主分割头、权重覆盖率和预处理，再比较 mIoU。源码里的空 OHEM、损失开关和恢复最佳模型逻辑需要修正后再用于新训练对照。

## 模型：P 保细节，I 聚合上下文，D 提供边界门控

| 变体 | `(m,n,C)` | PPM 中间通道 / 语义头中间通道 | D 路 stage 3 → stage 4 输出通道 | I 聚合 / 最终融合 |
| --- | --- | --- | --- | --- |
| S | (2,3,32) | 96 / 128 | 32 → 64 | PAPPM / Light_Bag |
| M | (2,3,64) | 96 / 128 | 64 → 128 | PAPPM / Light_Bag |
| L | (3,4,64) | 112 / 256 | 128 → 128 | DAPPM / Bag |

参数来自两个工厂 [models/pidnet.py:184–225](../../models/pidnet.py#L184)；D 路、PPM、融合选择见 [models/pidnet.py:57–85](../../models/pidnet.py#L57)。默认双卷积 stem 到 1/4，`layer1` 保持 1/4，`layer2` 到 1/8；P、D 自此保持 1/8，I 的 `layer3/4/5` 依次到 1/16、1/32、1/64，通道为 `4C/8C/16C`。P 的 `layer3_/4_` 均为 `m` 个 BasicBlock，I 的 `layer3/4` 均为 `n` 个 BasicBlock；P/D 末级瓶颈各输出 `4C`，PPM 也将 I 投影到 `4C` 后上采样至 1/8。[models/pidnet.py:23–55](../../models/pidnet.py#L23)、[136–175](../../models/pidnet.py#L136)。

- **PagFM（I→P）**：在两级分别把 I 压为 `2C`，再把 P、I 的投影逐通道相乘并求和，得到单通道空间门 `s=sigmoid(sum(q_I*k_P))`，输出 `(1-s)P+s*I_up`。P 不直接降分辨率。[models/pidnet.py:147–164](../../models/pidnet.py#L147)、[models/model_utils.py:268–312](../../models/model_utils.py#L268)。
- **I→D**：在 I 的 1/16、1/32 级各做 3×3 卷积与 BN、插值到 1/8 后相加到 D；`diff3/diff4` 是学习到的投影，不是固定差分算子。D 的末级多通道特征进入 Bag；单通道 D 头只供训练监督。[models/pidnet.py:57–92](../../models/pidnet.py#L57)、[147–180](../../models/pidnet.py#L147)。
- **PPM / Bag**：两种 PPM 均用原尺度、三档平均池化及全局池化，再拼接压缩并加 shortcut。DAPPM 逐级加前一尺度结果；PAPPM 四档各加 scale0，再以 4 组卷积并行处理。最终 `a=sigmoid(D)`：L 的 Bag 用 3×3 投影 `aP+(1-a)I`；S/M 的 Light_Bag 将两个 1×1 投影 `f_p(P+(1-a)I)` 和 `f_i(I+aP)` 相加。[models/model_utils.py:114–265](../../models/model_utils.py#L114)、[314–334](../../models/model_utils.py#L314)、[363–377](../../models/model_utils.py#L363)。
- **输出契约**：`get_seg_model` 固定 `augment=True`，即使 `.eval()` 也返回 `[P 辅助语义, 融合主语义, D 边界]`，通道数为 `[K,K,1]`，分辨率为输入的 1/8；`get_pred_model` 固定 `augment=False`，只返回融合主语义张量。P/D 辅助特征分别接自第一次 PagFM 和第二次 D 残差加法；其引用随后受到共享 `ReLU(inplace=True)` 的原地激活。头本身不放大，外层再按标签或输入尺寸上采样。[models/pidnet.py:33](../../models/pidnet.py#L33)、[87–92](../../models/pidnet.py#L87)、[147–182](../../models/pidnet.py#L147)、[218–225](../../models/pidnet.py#L218)、[models/model_utils.py:89–112](../../models/model_utils.py#L89)。

## 数据与有效训练配置

| 数据集 | train / val / trainval / test | 名单字段与标注 | 自带 YAML 的 train → 训练中 val |
| --- | --- | --- | --- |
| Cityscapes | 2975 / 500 / 3475 / 1525 | train/val/trainval 两列，`leftImg8bit` + `gtFine_*_labelIds.png`；test 一列图像 | 常规 S/M/L：train → val；trainval S/M/L：trainval → val |
| CamVid | 367 / 101 / 468 / 233 | 所有名单两列，`images/*.png` + `labels/*_L.png` | S/M：trainval → val；有标签的 test 需要显式切换 |

实读 [Cityscapes 名单](../../data/list/cityscapes) 和 [CamVid 名单](../../data/list/camvid)：各名单内部图像路径无重复；两套 `trainval` 都精确等于 `train ∪ val` 的记录集合，`train∩val=0`、`trainval∩test=0`，`trainval∩val` 分别为 **500/101**。这是路径与记录级检查，未查像素内容重复、文件齐备或相邻视频帧的相关性。配置示例见 [Cityscapes train:11–20](../../configs/cityscapes/pidnet_small_cityscapes.yaml#L11)、[Cityscapes trainval:11–20](../../configs/cityscapes/pidnet_large_cityscapes_trainval.yaml#L11)、[CamVid:11–20](../../configs/camvid/pidnet_small_camvid.yaml#L11)。README 的 CamVid 名单来源与数据划分提醒见 [README.md:79–89](../../README.md#L79)；这里只用仓库实际名单。

- Cityscapes 用 OpenCV 读 BGR 图像和灰度原始 labelId，显式把 19 个有效 ID 映射为 0–18，已列出的 void ID 映射到 ignore；列表路径是否含 `test` 同时控制一列解析和无标签返回。CamVid 用 PIL 读 RGB，按 11 个精确 RGB 颜色映射到 0–10，未匹配颜色赋 ignore。两者最终以 RGB / 255，按 `[.485,.456,.406]/[.229,.224,.225]` 标准化并转 CHW。Cityscapes 将固定 19 类权重直接传给语义损失，CamVid 权重为 `None`。[datasets/cityscapes.py:39–114](../../datasets/cityscapes.py#L39)、[datasets/camvid.py:36–95](../../datasets/camvid.py#L36)、[datasets/base_dataset.py:15–47](../../datasets/base_dataset.py#L15)、[tools/train.py:123–136](../../tools/train.py#L123)。
- 边界先由**已映射标签**做 `Canny(0.1,0.2)`，Cityscapes 丢弃四周 6 像素的 Canny 响应并补零，随后 4×4 核膨胀后二值化；CamVid 不做前述 6 像素边框处理。先生成边界再缩放；训练默认以长边基准乘 `0.5 + randint(0,16)/10`，即 0.5–2.1 共 17 档，图像用线性插值、标签/边界用最近邻；不足 crop 的右/下边先填图像 0、标签 ignore、边界 0，再随机裁剪；三者共享一次水平翻转。图像填 0 发生在归一化前。关闭 `TRAIN.MULTI_SCALE` 同时跳过随机裁剪/补齐，`TRAIN.FLIP` 独立。[datasets/base_dataset.py:49–133](../../datasets/base_dataset.py#L49)、[datasets/cityscapes.py:107–114](../../datasets/cityscapes.py#L107)、[datasets/camvid.py:87–95](../../datasets/camvid.py#L87)。
- 8 份 YAML 合并值：Cityscapes 默认宽×高 crop 1024×1024、base 2048、LR .01、配置 484 轮；CamVid 960×720、base 960、LR .005、配置 200 轮（实际迭代见下文）。默认每 GPU 6 张、两张可见 GPU，SGD momentum .9、WD .0005、Nesterov false；总 batch = 每 GPU batch × GPU 数，训练 `drop_last=True`。SGD 只有一个参数组、每 batch 在 `optimizer.step()` 后设下一步 poly LR `lr₀(1-t/T)^.9`。Cityscapes 从 ImageNet 初始化；CamVid 配置从对应 Cityscapes test 分割权重迁移。应随试验记录 seed、确定性设置和可见 GPU：入口默认种子 304，显式调用 Python `random.seed` 和 `torch.manual_seed`，YAML cuDNN benchmark=true、deterministic=false。[Cityscapes 示例:1–45](../../configs/cityscapes/pidnet_small_cityscapes.yaml#L1)、[CamVid 示例:1–45](../../configs/camvid/pidnet_small_camvid.yaml#L1)、[tools/train.py:37–102](../../tools/train.py#L37)、[138–182](../../tools/train.py#L138)、[utils/function.py:43–64](../../utils/function.py#L43)、[utils/utils.py:154–160](../../utils/utils.py#L154)。

| 配置 | 入口中的有效语义 |
| --- | --- |
| `TRAIN.BATCH_SIZE_PER_GPU/SHUFFLE/BASE_SIZE/IMAGE_SIZE/SCALE_FACTOR/MULTI_SCALE/FLIP`、`WORKERS` | 训练按字段接线，裁剪仅在多尺度开启时发生；`WORKERS` 也用于训练中验证，独立 eval 固定 0。[tools/train.py:82–121](../../tools/train.py#L82)、[tools/eval.py:83–99](../../tools/eval.py#L83) |
| `MODEL.ALIGN_CORNERS` | 默认 true，改变损失/评估外层插值；模型内部 PagFM、PPM、D 注入固定 false，custom 外层固定 true。[configs/default.py:29–33](../../configs/default.py#L29)、[models/pidnet.py:11–13](../../models/pidnet.py#L11)、[models/model_utils.py:8–10](../../models/model_utils.py#L8)、[298–309](../../models/model_utils.py#L298)、[utils/utils.py:41–46](../../utils/utils.py#L41)、[tools/custom.py:91–94](../../tools/custom.py#L91) |
| `MODEL.NUM_OUTPUTS` / `TEST.OUTPUT_INDEX` | 不决定工厂的 head 数。8 份 YAML 都是 2 / 1；训练验证把 2 解释成语义输出数，独立 eval 用 1 直接索引三元素列表。[utils/function.py:77–128](../../utils/function.py#L77)、[datasets/base_dataset.py:136–150](../../datasets/base_dataset.py#L136) |
| `TEST.IMAGE_SIZE/BASE_SIZE/BATCH_SIZE_PER_GPU` | 前两者传给数据集，但两条验证入口均 `multi_scale=False, flip=False`，当前数据集保持原尺寸；batch 字段仅训练中验证使用，独立 eval 固定 1。[tools/train.py:105–121](../../tools/train.py#L105)、[tools/eval.py:82–99](../../tools/eval.py#L82)、[datasets/base_dataset.py:106–133](../../datasets/base_dataset.py#L106) |
| `TEST.FLIP_TEST/MULTI_SCALE`、`PIN_MEMORY`、`AUTO_RESUME` | 测试增强字段无消费者；loader 的 pin 值写死（训练和其验证 false，独立 eval true）；恢复看 `TRAIN.RESUME`。[configs/default.py:14–20](../../configs/default.py#L14)、[79–88](../../configs/default.py#L79)、[tools/train.py:96–121](../../tools/train.py#L96)、[156–167](../../tools/train.py#L156)、[tools/eval.py:82–99](../../tools/eval.py#L82) |
| `LOSS.CLASS_BALANCE`、`TRAIN.BEGIN_EPOCH/EXTRA_EPOCH/EXTRA_LR`、`DATASET.EXTRA_TRAIN_SET` | 搜索运行代码只见声明，无控制路径；Cityscapes 固定权重，epoch 从 0 或 checkpoint 起，无 extra 阶段。[configs/default.py:36–77](../../configs/default.py#L36)、[tools/train.py:123–132](../../tools/train.py#L123)、[152–184](../../tools/train.py#L152) |

YAML 后合并命令行 `opts` 并冻结；可打印一个字段只证明它被解析。现有增强、类权重和测试入口实际行为需随基线锁定。[configs/default.py:91–97](../../configs/default.py#L91)。

## 损失与评估协议

模型外的 `FullModel` 将三路 logits 插值到标签大小。8 份 YAML 的默认配置均为 ignore=255、OHEM 打开、`OHEMTHRES=.9`、`OHEMKEEP=131072`、`BALANCE_WEIGHTS=[.4,1.]`、`SB_WEIGHTS=1.`。总损失实际为：

`0.4 × CE(P) + 1.0 × OHEM(主头) + 20 × 加权 BCE(D, 边界) + 1.0 × OHEM(主头, 预测边界选中像素)`。

OHEM 从有效标签的真类 softmax 概率升序取 `min(K,N-1)` 这个零基下标，再用严格小于 `max(该概率, .9)` 的像素取 CE 平均；辅助 CE 在 OHEM 模式下是未归约的图，标量项广播后由训练/验证端 `.mean()` 聚合（辅助项的 ignore 零值仍在均值的分母）。D 的 BCE 正/负像素分别乘负/正样本占比，对整个平坦化批次求平均；预测 D logits 经 sigmoid 后 **> .8** 才让对应语义标签进入 SB 项，其余赋 ignore。比较产生布尔掩码，SB 项不会经这条掩码给 D logits 传梯度；D 另受 BCE 和融合主头训练信号。`20` 和 `.8` 都没有 YAML 开关。DataParallel 各副本独立计算损失和 OHEM 候选，再汇总。[utils/utils.py:37–57](../../utils/utils.py#L37)、[utils/criterion.py:43–132](../../utils/criterion.py#L43)、[tools/train.py:123–136](../../tools/train.py#L123)、[utils/function.py:36–49](../../utils/function.py#L36)。

训练中验证保留辅助头并计算训练式损失，累积 P/主头各自的混淆矩阵，返回最后一个语义头（默认主头）的 mIoU 用于 best。独立评估也建三输出模型，数据集从 `OUTPUT_INDEX` 选一头并把 logits 上采样到输入尺寸，然后 `.exp()`；结果未做概率归一化。没有浮点并列和二次插值时，指数变换不改变 argmax。带标签 `testval` 累加混淆矩阵，按 `TP/max(1,TP+FP+FN)` 对固定类别数做宏平均（缺席类别贡献 0），报告全局 pixel accuracy 和类别平均 accuracy；忽略配置的 void。两条入口都是单尺度、无翻转。[utils/function.py:77–177](../../utils/function.py#L77)、[datasets/base_dataset.py:136–150](../../datasets/base_dataset.py#L136)、[utils/utils.py:129–152](../../utils/utils.py#L129)。

Cityscapes test 由 `tools/eval.py` 判定后仅导出 PNG 至 `test_results`，预测 trainId 逆映射为原始 labelId，按图像名保存；仓库没有在这条路径上计算 test mIoU。CamVid `test.lst` 有标签，走 `testval`，README 的示例显式传入名单。Cityscapes README 的 val/test 指标引用不同权重，并明确 Cityscapes test 的命令用于生成结果；本轮未重测这些指标。训练中验证必须使用有标签的名单，否则 Cityscapes test 的三项数据不符合其五项数据接口。[tools/eval.py:104–121](../../tools/eval.py#L104)、[utils/function.py:180–202](../../utils/function.py#L180)、[datasets/cityscapes.py:84–127](../../datasets/cityscapes.py#L84)、[README.md:60–71](../../README.md#L60)、[102–121](../../README.md#L102)。

## 问题、影响及最小验证

| 判定 | 触发与影响 | 最小修复建议 / 轻量验收 |
| --- | --- | --- |
| **已确认：trainval 的 val 重叠** | 三份 Cityscapes trainval 和两份 CamVid YAML 将 trainval 用于训练并在 val 上选 best；500/101 张 val 全部在训练名单内。该读数不能代表独立泛化；独立 test 并不与训练列表重叠。 | 常规实验分开 train/val；保留 trainval 发布配方时，将样本内 val 显式标注，并预先固定训练/选权重策略，最终只按独立 test 协议报告。验收按图像键做交集并记录名单 SHA/计数；不把 README 的独立 val/test 表误当作该问题的证据。 |
| **已确认：权重可能静默少载** | 非 ImageNet 的训练初始化和 custom 对所有 key 无条件切 6 字符后按名/形状匹配，eval 也切 6 字符但不筛形状；例如裸 `conv1.0.weight`、`module.` 和第三方命名不匹配时，已先用初始化字典填满，eval 最后的 strict load 仍可通过并继续评分。内存 mock 中 custom 加载裸 key 或 `module.` 后参数仍为初值 1，只有 `model.` 更新为 9。eval 还先要求 YAML 的 `MODEL.PRETRAINED` 可作 ImageNet 格式读取，即使 `TEST.MODEL_FILE` 已提供，目标权重的 `torch.load` 也未指定 map_location。实现位置：[models/pidnet.py:193–214](../../models/pidnet.py#L193)、[tools/eval.py:58–78](../../tools/eval.py#L58)、[tools/custom.py:59–72](../../tools/custom.py#L59)。本报告未判定任何现有权重的实际覆盖率。 | 用统一 loader 识别受支持的前缀和格式，显示缺失/多余/错形状及主头/各分支覆盖率；独立 eval 从无预训练构造并要求除明确豁免项外完整匹配，迁移模式才允许声明的分类头不匹配。用同一合成 state_dict 的裸 key、`model.`、`module.` 及错误类数验证拒绝和匹配行为。 |
| **已确认：OHEM 空集合可崩溃或出 NaN** | 主项全 ignore，或 SB 项没有 `sigmoid(D)>.8` 的有效语义像素（包括全是 void、某副本单独没有像素）时，`pred[-1]` 访问空张量；有候选但严格阈值排光同分像素时，空均值为 NaN，训练循环未检查有限性。CPU 的完整 `FullModel` 上用正常语义标签、边界 logits=0 得 `IndexError`；两个真类概率都约 .993 得 NaN。[utils/utils.py:52–57](../../utils/utils.py#L52)、[utils/criterion.py:63–78](../../utils/criterion.py#L63)、[utils/function.py:43–49](../../utils/function.py#L43)。 | 无有效像素时返回有限且与图相连的零损失；确保非空候选至少保留 `min(K,N)`（例如阈值集合与排序前 K 索引取并）。保留现有已满足保留量的候选集合，验收全 ignore、SB 空、单有效像素、同分、普通样本及反传。 |
| **已确认：关闭 OHEM 的配置不可用** | `LOSS.USE_OHEM=False` 改用 CrossEntropy；SB 项传的是单个 BCHW 张量，CE 却按 `len(score)` 把 batch 当输出头，普通批量失败；`MODEL.NUM_OUTPUTS=1` 也无法把真实多头列表改成单头。CPU 的完整 `FullModel` 复现形状错误。[tools/train.py:123–132](../../tools/train.py#L123)、[utils/utils.py:48–54](../../utils/utils.py#L48)、[utils/criterion.py:25–38](../../utils/criterion.py#L25)。 | 根据输入类型把张量包装为单元素列表，显式断言输出/权重契约；普通 CE 的全 ignore 也要有有限零损失。验收 batch 1/2/>2 的主项 + SB 项，检查权重和反传。 |
| **已确认：训练 Acc 将 void 算作错分** | `pixel_acc` 用 `label >= 0`，8 份 YAML 的 ignore=255 会计入分母；两个像素（1 个有效且预测对、1 个 void）训练 Acc=0.5，混淆矩阵法=1.0。独立评估 mIoU 已按配置排除 void。[utils/utils.py:29–35](../../utils/utils.py#L29)、[129–152](../../utils/utils.py#L129)。 | Acc 用与语义损失一致的 ignore 过滤；合成全有效/混合/全 ignore 样本验证，并保留独立评估口径。 |
| **已确认：最佳分数的恢复元数据滞后一轮验证** | 先把旧 `best_mIoU` 写入 checkpoint，再比较本轮指标、更新 `best.pt`。若在一次提升后恢复，介于旧分数和真实最佳之间的结果可覆盖更好的 `best.pt`。未运行验证的轮次还继续打印上一轮 loss/mIoU。默认 Cityscapes 从第 0 轮起前段每 5 轮验证、最后 100 轮每轮验证；恢复入口首轮强制验证。[tools/train.py:154–205](../../tools/train.py#L154)。 | 只在本轮实际验证时更新 best 并写日志，之后再保存最新 checkpoint 的 best 元数据。用合成指标序列 .70 → .80 → 重启 → .75 检查 best 留在 .80；检查跳过验证轮无新指标。 |
| **已确认：CamVid 轮数与 LR 调度分离** | 根据 `'camvid' in TRAIN_SET` 将循环固定为索引 0–120（从头 121 轮）；poly 的总步数仍用 YAML 的 200 轮。按 468 张、batch 12 算得每轮 39 步，调度函数在最后一步 `t=4718,T=7800` 实测设置 LR=.002167875，仍为起始的 43.36%；修改 `END_EPOCH` 不会改变停止轮数，配置过小还可能使 poly 底数为负。[tools/train.py:169–184](../../tools/train.py#L169)、[utils/utils.py:154–160](../../utils/utils.py#L154)。 | 把停止轮数与调度长度显式配置并校验；历史复现先保留 121/200，若意图一致则统一并另做对照。用小步数 mock 检查循环边界、每步 LR 和恢复索引。 |
| **已确认：输出选择容易静默错评** | 正常 YAML 都显式用索引 1；若依赖默认 `-1` 则选择单通道 D，argmax 恒为 0；索引 0 可成功给出 P 的 mIoU。把 `MODEL.NUM_OUTPUTS` 改为 1 也不会去掉实际头，独立 eval 可能将列表传给插值，训练验证仍只分配一张混淆矩阵。[configs/default.py:29–33](../../configs/default.py#L29)、[79–88](../../configs/default.py#L79)、[datasets/base_dataset.py:136–150](../../datasets/base_dataset.py#L136)、[utils/function.py:77–106](../../utils/function.py#L77)。 | 指定主头接口并断言 BCHW、通道数等于类别数；三头训练路径校验期望数量。用头值不同的 fake 模型检查主头、D 头拒绝和推理张量路径。 |
| **已确认：模型尺寸和 Cityscapes 导出的备用分支有边界问题** | 内部融合用输入 `H//8,W//8`，卷积的实际 1/8 特征按向上取整；合成 64×81 对三变体均在 D 加法报错，正常 64×80 正常。若 Cityscapes test 预测与原图尺寸不同，备用插值从 HWC 的 `size[-2:]` 取到 W,C，且 collate 后是 Tensor 尺寸；现有 Torch 1.13 CPU stub 实测 TypeError，改成 Python 整数后仍是 W,C。当前默认单尺度先插值到原图，正常情况不触发备用分支。[models/pidnet.py:136–152](../../models/pidnet.py#L136)、[datasets/cityscapes.py:94–105](../../datasets/cityscapes.py#L94)、[utils/function.py:185–196](../../utils/function.py#L185)。 | 内部按实际共享特征 `shape[-2:]` 对齐，或入口 pad 到 8 倍数后裁回；备用导出用显式 `(int(H),int(W))`。回归 64×80 数值不变、64×81 形状正确，另用 H=3,W=5,C=3 的 stub 检查输出大小。 |
| **已确认：默认 `IGNORE_LABEL=-1` 与 uint8 管道不兼容** | 8 份 YAML 都改成 255，因此现有配方未触发；自建配置沿用默认 -1 时，灰度 uint8 映射 -1 实测变为 255，cv2 以 -1 补 uint8 实测变成 0，导致前者不被排除、后者混入 0 类。[configs/default.py:68](../../configs/default.py#L68)、[datasets/cityscapes.py:84–109](../../datasets/cityscapes.py#L84)、[datasets/camvid.py:66–71](../../datasets/camvid.py#L66)、[datasets/base_dataset.py:46–67](../../datasets/base_dataset.py#L46)。 | 把默认统一为 255 并约束 uint8 标签的 ignore 范围，或者引入独立 signed 标签表示；用 raw 0/7、pad 和混淆矩阵验证 void 全程一致。 |
| **已确认：CLI 会误导调试** | eval 默认 `experiments/...` 路径不存在，需显式 `--cfg`。custom 的 `--c False` 被 Python 转成 True，始终使用 Cityscapes 19 色表；POSIX 下 `split("\\")` 不提取 basename，`samples/demo.png` 会尝试从 `samples/samples/demo.png` 读取。模型工厂用名字是否含 s、m 判断，未知名称可能静默选错；现有 YAML 名均匹配。[tools/eval.py:27–40](../../tools/eval.py#L27)、[tools/custom.py:39–46](../../tools/custom.py#L39)、[74–103](../../tools/custom.py#L74)、[models/pidnet.py:184–191](../../models/pidnet.py#L184)、[218–225](../../models/pidnet.py#L218)。 | eval 默认路径修为 `configs/...`；custom 使用显式数据集/模型枚举和相应 palette、`Path`/basename，并检查读图；验证 POSIX 相对/绝对路径、CamVid 类数/颜色和未知型号报错。 |
| **设计/风险：边界监督含 void 转换，BCE 纯单类无梯度** | Canny 未屏蔽 void，只有 0↔255 交界的 24×24 合成样本仍有 96 个正边界像素（关闭 Cityscapes 边框处理时）；padding 为边界 0。BCE 对全正或全负批次权重为零，实测 loss=0。真实数据上 void 交界对边界质量和最终 mIoU 的贡献未知。[datasets/base_dataset.py:60–67](../../datasets/base_dataset.py#L60)、[106–119](../../datasets/base_dataset.py#L106)、[utils/criterion.py:102–119](../../utils/criterion.py#L102)。 | 基线保留；想改善时独立比较有效像素对上的类别边界、排除 void 邻域、单类批次处理，先检查边界正负比和 void 邻域比例，再看边界指标与主 mIoU。 |

其他工程限制：[tools/train.py:70–77](../../tools/train.py#L70) 要求配置 GPU 数等于可见设备数，数量不符返回 0 并不表示训练完成；Cityscapes 构造时类权重直接 `.cuda()`，脚本入口也使用 CUDA。训练模式下 PPM 对 1×1 全局池化结果做 BN，每副本 batch=1 可能因没有足够样本而报错；CPU 推理形状检查均以 `.eval()` 执行。[datasets/cityscapes.py:55–59](../../datasets/cityscapes.py#L55)、[models/model_utils.py:133–137](../../models/model_utils.py#L133)、[215–219](../../models/model_utils.py#L215)。代码使用 `np.int`，应记录实际依赖版本；本次 NumPy 1.21.6 下混淆矩阵可执行，移除了该别名的环境会触发兼容性问题。[datasets/base_dataset.py:78–87](../../datasets/base_dataset.py#L78)、[utils/utils.py:129–136](../../utils/utils.py#L129)。

## 复现与未决

CPU 使用仓库根目录、`/home/robot/anaconda3/envs/segformer/bin/python -B`，实测 Torch 1.13.0 / NumPy 1.21.6 / OpenCV 4.1.2。下列最小片段对应已执行的合成检查，未调用 CLI 主训练或评估流程：

```bash
/home/robot/anaconda3/envs/segformer/bin/python -B - <<'PY'
from pathlib import Path
import torch
from models.pidnet import PIDNet
from utils.criterion import OhemCrossEntropy
from utils.utils import FullModel, get_confusion_matrix

for ds in ('cityscapes', 'camvid'):
    rows = {p.stem: [tuple(l.split()) for l in p.read_text().splitlines()]
            for p in Path('data/list', ds).glob('*.lst')}
    keys = {n: {r[0] for r in rs} for n, rs in rows.items()}
    print(ds, {n: len(rs) for n, rs in rows.items()},
          'val overlap:', len(keys['trainval'] & keys['val']))
    assert set(rows['trainval']) == set(rows['train']) | set(rows['val'])
    assert not keys['trainval'] & keys['test']

torch.set_num_threads(1)
for args in ((2, 3, 19, 32, 96, 128), (2, 3, 19, 64, 96, 128),
             (3, 4, 19, 64, 112, 256)):
    for augment in (True, False):
        model = PIDNet(*args, augment=augment).eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 3, 64, 80))
            shape = [tuple(x.shape) for x in out] if augment else tuple(out.shape)
            print(args[:2], args[3], augment, shape)
            assert shape == ([(1, 19, 8, 10), (1, 19, 8, 10), (1, 1, 8, 10)]
                             if augment else (1, 19, 8, 10))
            try:
                model(torch.zeros(1, 3, 64, 81))
            except RuntimeError as e:
                assert 'must match' in str(e)
            else:
                raise AssertionError('expected odd-width mismatch')

logits = torch.tensor([[[[5., 5.]], [[0., 0.]]]])
ohem = OhemCrossEntropy(ignore_label=255, thres=.9, min_kept=131072)
try:
    ohem(logits, torch.full((1, 1, 2), 255, dtype=torch.long))
except IndexError:
    pass
else:
    raise AssertionError('expected empty OHEM crash')
assert torch.isnan(ohem(logits, torch.zeros(1, 1, 2, dtype=torch.long)))
labels = torch.tensor([[[0, 255]]])
cm = get_confusion_matrix(labels, logits, labels.size(), 2, 255)
assert FullModel(None, None, None).pixel_acc(logits, labels) == .5
assert cm.trace() / cm.sum() == 1.
PY
```

此外已逐个合并 8 份 YAML 并核对模型工厂分派、输出索引、输入尺寸、ignore、LR 和损失字段；用 `FullModel` stub 检查了空 SB 和普通 CE 错误，用假 dataset/无写盘的 `test` stub 核对错尺寸导出；直接调用 CLI 参数解析和标签/边界函数核对表内触发现象。尚未衡量这些问题对现有训练日志和真实 mIoU 的频率/幅度；现有本地权重的 key 覆盖、真实数据完整性、外部基准协议与改造增益由其他验证或后续实验提供。
