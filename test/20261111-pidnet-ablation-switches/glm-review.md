# PIDNet GLM 审查与修复记录

> **Agents:**
> - sol-lead / pidnet_ablation_lead（gpt-6.1-sol/max）— 审查证据与修复记录
> - DeepSeek（opencode-go/deepseek-v4.1-flash/high）— 独立会话提供润色提案
> **Date:** 2026-11-11
> **Experiment:** 20261111-pidnet-ablation-switches

原实现工单 `27dc31ff1e264492b988d20ebb9235c9`。补充审查工单 `8a66db15b2314c64a85fdc1effa19c2b` 的两次 Super Agent `review-run` 各约 90 秒超时，均未返回报告；原生尝试因宿主 `unreadable_encrypted_agent_task` 未开始。经用户授权，同模型在独立 App Server 会话约 571 秒后返回有效报告。宿主回执模型 `zai/glm-5.3`、档位 `high`；只读拒写探针通过、网络和工具禁用，工具调用数为 0。回执与原意见存于 `.super-agent/authorized-glm-review-result.json`；此结果不记为前两次插件审查通过。

| GLM 发现 | 实际修复与验证 |
| --- | --- |
| minor：fixed 和 FADC 的率不入 state_dict，旧 strict 加载可在错率下成功。 | `models/pidnet.py` 给非 none 模型加入由 mode/stage/block/rate 编码的持久标记键。不同几何下，工具和 PyTorch 原生 strict 加载均拒绝；部分预训练只取兼容键，保留目标模型实际率与标记。默认模型无新增键。实测 [CPU 8 项](logs/context-post-review.log) 与 [CUDA 7 项](logs/atrous-post-review.log)；[合成 CLI](logs/verify-ablation-post-review.log) 同设置通过，实际 eval 在配置率为 4、加载率为 2 的权重时明确失败。 |
| note：FreqSelect 注释把残差增益的单位点写成恒等。 | `models/fadc.py` 更正为零残差增益还原输入，算法未改。[频带与梯度测试](logs/atrous-post-review.log) 仍通过。 |

GLM 原报告核对了代码和提交的部分日志，未独立运行 GPU 或真实数据评估。后续同一模型和档位的独立只读会话定向复核两项修复，返回 `pass`、两项 resolved、无新增发现；快照未变，工具调用数为 0。完整回执在 `.super-agent/authorized-glm-fix-review-result.json`。

合成组合的唯一训练步损失为 **14.690834**（语义 **6.061063**、边界 **4.215402**、BAS **4.414369**）。后续 `model.eval()` 的合成验证总损失为 **7575.002**（初次 **7575.012**），有限但偏大。日志未分解验证损失，也没有同条件基线，原因未定位；这些数据只能验证入口和损失能运行，不能说明收敛、稳定性或真实精度。

修订：2026-11-11，sol-lead 记录 GLM 对两项修复的定向复核结果。

修订：2026-11-11，sol-lead 对照已核实原稿采用 DeepSeek 的措辞提案；[报告](report.md)注明原生失败与同模型会话来源。
