# PIDNet-L 单卡 RTX 4090 训练 — 运行计划

> **Agent:** deepseek-writer（模板）与 train_pidnet_l_4090.py（运行填充）
> **Date:** {date}
> **Experiment:** {experiment}

## 目标与假设

- 目的：在单张 RTX 4090 桌面版（排除 4090D 与笔记本机型）上，按 PIDNet-L 既有配方训练本实验指定的数据集与预设，并留下可追溯记录。
- 预期：预检项全部通过后启动训练；精度、耗时与显存不作为本计划的承诺值。

## 基线与输入

{metadata}

以上由运行脚本在启动前写入；缺失项须标注为缺失，不以估计值代替。

## 改动点

- 模型结构、损失、优化器与日程沿用 PIDNet-L 既有配置；本 run 只覆盖数据集、预设、GPU、batch、workers 与显式给出的 I 位置选项。
- --stage 与 --block 仅在显式给出时生效；默认保持旧 L 逻辑的 stage 5、block 1、conv2。
- 不改 models/、utils/、datasets/、tools/train.py 与基线 configs；如记录显示存在其他改动，本计划须按实际情况修订并追加修订记录。

## 数据与划分

- Cityscapes：19 类，train.lst 2975 行 → val.lst 500 行。
- CamVid：11 类，train.lst 367 行 → val.lst 101 行。
- 训练只用 train.lst，验证只用 val.lst；test.lst 不参与训练、选 checkpoint 或阈值选择。
- 实际名单路径与 SHA-256 见上文记录；自定义数据根须提供同构名单与图像、标签。

## 运行命令

计划命令如下；实际执行的命令以脚本打印与 logs 记录为准：

~~~bash
{command}
~~~

## 评估指标与成功标准

- 运行判据：预检通过、训练进程退出码可追溯、checkpoint 与日志落在本 run 目录。
- 验证协议：单尺度、无 flip；不采集或承诺 mIoU、训练时长与显存峰值。
- 失败判据：任一预检项不符即不启动；中途失败保留退出码、日志与耗时，不做静默重试，也不自动改 batch、epoch 或学习率。

## 资源与时间预算

- 单张 RTX 4090 桌面版，FP32，每卡 batch 默认 2；显存与耗时未在本机实测，不作为承诺。
- OOM、数据缺失、权重不匹配、兼容检查失败即为本次运行的失败边界；失败后停止并记录。

## 风险与回滚

- 权重不匹配时停止；CamVid 须显式提供可信的 19 类 City-L 权重。
- run 目录已存在或非空时拒绝覆盖，改用新的 run 名。
- 回滚只针对本 run 目录，删除前须确认归属。

## 修订记录

- 模板：deepseek-writer，{date}，初版。运行填充后的修订由执行者在本节追加。
