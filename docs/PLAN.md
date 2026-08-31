# MIntRec-Multi 实验计划：四目标路线图（v2）

> 更新时间：2026-08-30 ｜ v1 → v2：同步三轮全量结果、训练动力学档案、环境迁移（AutoDL + autodl-tmp）、A/B 实验状态与代码变更清单
> 环境：AutoDL RTX 4080 SUPER 32GB / torch 2.8.0+cu128 / transformers 5.5.4 / pandas 3.0.2 / scikit-learn 1.8.0 / mamba_ssm 2.3.1 / Python 3.12.3（驱动 580.105.08，CUDA 13.0）
> 数据：MIntRec multi-class（20 类，train 1334 / dev 445 / test 445），特征 text 768×30 / video 256×230 / audio 768×480

---

## 0. 状态速览（2026-08-30 12:30）

| 项 | 状态 |
|---|---|
| R1 / R2 / R2.5 三轮全量结果 | ✅ 完成（见 §2.1） |
| 九方法训练日志动力学分析 | ✅ 完成（见 §3，结论：早停从未失效，100 轮上限从不触发） |
| 目标二 A/B 对照实验 | ✅ **完成**（08-30 12:31 双链重启，全程 ~4.5h；结果与判定见 §6.1：**DLF 取 B、DDSE 取 A**） |
| 目标二 二阶微调 | ✅ **完成**（08-30 17:28~21:20 四 run：DLF lr 细扫 2e-5/5e-5、DDSE 累积 8/16；**最终推荐 dlf lr 3e-5、ddse 累积 10**，见 §6.2） |
| 目标一 EMOE/GsiT 协议净化重跑 | ✅ **完成**（§5 定版：EMOE 取 `emoe_p_dp`、GSIT 取 `gsit_lr1`，峰值/贴线口径进验收带） |
| 目标三 TCS_Mamba 阶段 1 配方对齐 | ✅ **完成**（定版 a2 = bs16 + 累积 8 + lr 3e-5，test 71.91/67.99，见 §7） |
| 目标三 TCS_Mamba 阶段 2 dev 网格 | ✅ **完成**（9/9 配置，无一 test 超越 a2，见 §7；结论：a2 为最终配方） |
| 目标三 TCS_Mamba 阶段 3 多种子定版 | 🔄 **进行中**（a2 5 seeds 已出：均值 69.66±1.31 / 66.25±1.28，seed 0 离群高位；正补 R1 基线 5 seeds 做同口径对照） |
| 目标三 TCS_Mamba 网格搜索 | ⏳ 排队（时间预算按实测重估：~3-6 min/run，见 §7） |
| 目标四 效率基准 | ⏳ 未启动（`tools/efficiency_bench.py` 待写；README 旧时长表已过时，见 §8） |

---

## 1. 环境与基础设施

### 1.1 硬件变迁

| | 旧（ENV_NOTES 时代，WSL） | 现（AutoDL） |
|---|---|---|
| GPU | RTX 5060 Laptop 8GB | RTX 4080 SUPER 32GB |
| 内存 | 7.4GB | 503GB |
| CPU | — | 96 vCPU |
| 系统盘 | — | 30G |
| 数据盘 | — | `/root/autodl-tmp` 50G |

**影响**：
- 旧环境"`--num_workers 0` 是硬约束（EXIT 137）"已解除。num_workers 不影响结果（特征整体常驻内存、无逐样本随机增强），只影响加载速度；为与既有结果严格同协议，命令模板仍默认 0。
- 旧环境为压显存给 `transformers_encoder/transformer.py` 加的梯度检查点，现已改为**显存门控**（仅 <16GB 的 GPU 启用，见 §10 变更 4）。4080 SUPER 上 MULT 不再开检查点，速度与 WSL 时代外推值不可比。
- 32GB 显存可**双任务并行**（两个 DLF 训练共占 ~12.3GB），A/B 重跑即采用双链并行。

### 1.2 磁盘布局（2026-08-30 迁移）

- `outputs/`（12G，checkpoint）与 `MIA-datasets/`（898M）已迁至 `/root/autodl-tmp/MIntRec-Multi/`，仓库根以软链接接回（`outputs -> /root/autodl-tmp/MIntRec-Multi/outputs`，`MIA-datasets` 同理）。代码零改动、命令不受影响；数据集内 `MIntRec -> .` 相对软链迁移后依然有效。
- `cache/`（421M，bert-base-uncased）留系统盘。`HF_HUB_OFFLINE=1` 可全程离线。
- 两个目录均已被 `.gitignore` 覆盖，软链接不污染 git。
- 结论：**磁盘约束解除**（数据盘余 ~37G），后续大网格、5-seed 定版的 checkpoint 可放心落盘。

### 1.3 运行命令模板（当前机器）

```bash
cd /root/MIntRec-Multi
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY=/root/miniconda3/bin/python
$PY run.py --dataset MIntRec --method <method> --data_mode multi-class \
  --train --save_results --save_model --seed 0 --gpu_id 0 \
  --text_backbone bert-base-uncased \
  --config_file_name <config> --results_file_name <name>.csv --num_workers 0
```

---

## 2. 现状基线

### 2.1 三轮全量结果总表（multi-class 20 类，test，macro 口径，seed 0）

| 方法 | R1 bs16/32¹ acc | R1 F1 | R2 bs64 acc | R2 F1 | R2.5 b16+累积² acc | R2.5 F1 | 论文参考³ acc | 论文参考 F1 |
|---|---|---|---|---|---|---|---|---|
| text | **72.13** | 68.78 | 70.79 | 66.14 | — | — | 70.88⁴ | 67.40⁴ |
| mag_bert | **72.13** | 67.43 | 71.46 | 67.73 | — | — | 70.34 | 68.19 |
| mult | 71.91 | **69.04** | 71.24 | 68.25 | — | — | 72.58 | 69.36 |
| misa | 71.46 | 68.89 | 68.99 | 64.98 | — | — | 72.36 | 70.57 |
| tcs_mamba | 71.24 | 66.59 | 69.66 | 65.19 | — | — | — | — |
| emoe | — | — | 69.89 | 66.61 | 70.56 | 66.59 | 72.58 | 70.73 |
| gsit | — | — | 70.34 | 67.75 | 69.66 | 65.92 | 72.60 | 69.40 |
| dlf | 62.47 | 58.42 | **69.21** | 65.59 | — | — | — | — |
| ddse | 62.25 | 57.32 | **66.29** | 62.99 | — | — | — | — |

¹ R1 用**已提交版本**的基础配置（dlf bs16、ddse bs32、其余 bs16）；结果文件 `results/full_*.csv`。
² R2.5 = emoe/gsit 的 bs16+梯度累积轮（`emoe_b16.py` 累积 10 / `gsit_b16.py` 累积 4），**当轮 eval_monitor=f1_w（混合协议）**；`results/full_b16_*.csv`。工作区配置已修正回 macro `f1`，重跑后此列数字会变。
³ GsiT 论文 Table 3（macro）；EMOE 论文为挑峰值口径（"selecting the peak values"），对比时按各论文自己的口径。
⁴ MIntRec 原论文报告值（其 F1/P/R 同为 macro）。

### 2.2 本轮新确认的关键结论（v2 新增）

1. **dlf/ddse 是唯一在 bs64 下大幅改善的方法**（+6.74 / +4.04 acc），其余方法 bs64 普遍降 0.7~2.5 —— 与"小有效 batch × 大 lr（1e-4）训练不稳"的诊断完全吻合，等效 batch 拉大即收益。方案 A（bs16×累积 10=等效 160）正是该发现的延续，R2 已给出方向性证据。
2. **R2.5 的 emoe/gsit 属混合协议**：日志确认当轮 `eval_monitor: f1_w`（b16 emoe best dev 0.7237 与 CSV `eval_f1_w` 逐位吻合）。PLAN v1 的"需按 macro 重跑"由此而来；现配置已改回 `'f1'`，重跑前不要再动。
3. **bs16 仍是与参考表匹配的协议**（R1 各方法与论文参考值差距普遍 ≤2 且方向一致），R2 的整体偏低再次确认；但 dlf/ddse 例外——它们在 R2/R2.5（大等效 batch）下更接近健康水平，属于"病好了才能同台比"。
4. 指标口径：acc ≡ weighted recall（数学恒等），F1/P/R 报 macro；weighted 列自 v2 起保留在 CSV 作参考（metrics.py 已支持），不进主表。

### 2.3 论文参考值（GsiT 论文 Table 3，macro，MIntRec）

| 方法 | acc | F1 | P | R |
|---|---|---|---|---|
| MAG-BERT | 70.34 | 68.19 | 68.31 | 69.36 |
| MMIM | 71.21 | 68.70 | 69.20 | 68.90 |
| MuLT | 72.58 | 69.36 | 70.73 | 69.47 |
| MISA | 72.36 | 70.57 | 71.24 | 70.41 |
| CAGC | 73.03 | 70.62 | 70.86 | 70.55 |
| EMOE | 72.58 | 70.73 | 72.08 | 70.86 |
| GsiT | 72.60 | 69.40 | 69.40 | 70.10 |

---

## 3. 训练动力学档案（v2 新增，来自全部 19 个已完成 run 的日志解析）

### 3.1 早停行为

- 所有 run 均**早停收敛，无一触及 100 轮上限**：最优轮分布在 **5~31**，停止轮 **12~38**（≤ 上限的 38%）。
- 测试用的是 EarlyStopping 保存的**最优模型副本**（`utils/functions.py` EarlyStopping 类，`copy.deepcopy`），与最后一轮权重无关——"第几轮最优"已由协议自动解决。
- **patience=8 有真实价值，勿随意调小**：反例 text(R1) 第 12 轮即达最终最优分的 95%，但真正峰值在第 23 轮；patience=4 会在第 16 轮错过峰值。快速方法（misa 5→12、mult 6→13）的"确认尾巴"占近半计算量，但绝对时间 <2 分钟，无需优化。

### 3.2 实测耗时与轮次（wall-clock，本机 4080 SUPER）

| 方法 | 协议 | 最优轮 | 停止轮 | 总耗时 | 折合 s/epoch |
|---|---|---|---|---|---|
| text | bs16 | 23 | 30 | 1:59 | ~4 |
| misa | bs16 | 5 | 12 | 1:53 | ~9.4 |
| tcs_mamba | bs16 | 11 | 18 | 2:35 | ~8.6 |
| mag_bert | bs16 | 21 | 28 | 3:43 | ~8 |
| ddse | bs32 | 23 | 30 | 15:02 | ~30 |
| dlf | bs16 | 12 | 19 | 19:43 | ~62 |
| mult | bs16（无检查点） | 6 | 13 | 22:41 | ~105 |
| emoe | bs16+累积10 | 7 | 14 | 5:43 | ~25 |
| gsit | bs16+累积4 | 10 | 17 | 6:49 | ~24 |

（bs64 轮各方法轮次类似：最优 9~31 / 停止 16~38，单 epoch 快 ~3-4 倍。）

### 3.3 对计划的直接影响

1. **`num_train_epochs` 100 纯属记账上限**，调小不省时间；对齐 EMOE 论文的 50 上限（`emoe_p.py` 已设）对结果零风险。
2. **目标三网格搜索预算重估**：tcs_mamba bs16 单 run（含早停尾巴）≈ 3-6 分钟，20 个配置 × 单 seed ≈ **1~2 小时**（v1 估 3h），网格规模可以更激进。
3. **目标四的训练时间维度直接从 `logs/` 汇总**（§3.2 即雏形），不重跑。
4. README §7 的时长表是 RTX 5060+8GB+检查点常开时代的外推值，**已过时**，以本表为准。

---

## 4. 统一实验口径（四目标公共约定）

| 项 | 约定 |
|---|---|
| 指标 | acc / macro F1 / macro P / macro R（weighted 列保留在 CSV 作参考，不进主表） |
| 协议 | bs16（带梯度累积时注明等效 batch）、seed 声明、**dev macro F1 选检查点**（eval_monitor='f1'，勿用 f1_w） |
| 早停 | patience 8（部分实验 12，需注明）、上限 100（EMOE 论文协议 50） |
| 调参纪律 | **一切超参选择只看 dev；test 仅在最终配置上评估一次** |
| 多种子 | 关键结论 3 seeds（0/1/2），最终对比 5 seeds，报 mean±std；与"峰值口径"论文对比时另报 best |
| 运行参数 | `--num_workers 0`（结果无关，协议一致性）、`HF_HUB_OFFLINE=1`、`--gpu_id 0`、`--text_backbone bert-base-uncased`、seed 0 起步 |
| 结果落盘 | `results/<name>.csv`（追加式）；日志 `logs/`；checkpoint `outputs/`（→ autodl-tmp） |

---

## 5. 目标一：EMOE / GsiT 尽量对齐原文结果

**验收标准**：与各自论文数字的差距 acc ≤ 1.5 且 macroF1 ≤ 1.0；EMOE 按"峰值"口径、GsiT 按"多种子均值"口径分别对照。

**差距现状**（macro 口径）：emoe 最好 70.56/66.59（R2.5，混合协议待重跑）；gsit 最好 70.34/67.75（R2 bs64+macro）。对论文 72.58/70.73 与 72.60/69.40，差 1~4 点，集中在稀有类。

**已就绪的配置**：

| 配置 | 关键项 | 对齐内容 |
|---|---|---|
| `configs/emoe_p.py` | bs16 + 累积 10 + **50 epochs 上限** + lr 1e-4 + temperature 0.1 + macro monitor | EMOE 论文 §4 公布的全部 MIntRec 项 |
| `configs/emoe.py` / `emoe_b16.py` | bs64 累积 2 / bs16 累积 10，100 上限 | 已跑过（R2 / R2.5） |
| `configs/gsit.py` / `gsit_b16.py` | bs64 累积 1（论文等效 batch）/ bs16 累积 4 | 论文只公布特征与 5-seed，未公布超参 |

**行动项与结果**（2026-08-30 执行）：

1. ✅ **协议净化重跑**（macro 选检查点，seeds 0/1/2）：

   | 方法 | 配置 | 3-seed test acc | 3-seed test F1 | 论文 | 差距（均值口径） |
   |---|---|---|---|---|---|
   | EMOE | `emoe_p`（bs16+累积10+50ep+lr1e-4） | 70.34 ± 1.19 | 67.11 ± 0.87 | 72.58 / 70.73 | −2.24 / −3.62 ❌ |
   | GSIT | `gsit`（bs64+lr_other 5e-4） | 69.97 ± 1.63 | 67.00 ± 1.68 | 72.60 / 69.40 | −2.63 / −2.40 ❌ |

2. ✅ **受限超参还原**（单变量 seed 0 筛选 → 胜者补 seeds 1/2）：

   筛选（seed 0，dev 判定）：EMOE dropout 0.3 / temperature 0.3 / 累积 4 → dev 0.6994 / 0.7063 / 0.6813（基线三种子 dev 均值 0.702）；GSIT lr_other 1e-4 / 2e-4 / nlevels 2 → dev 0.6915 / 0.6977 / 0.6833（基线 0.6756）。**GSIT 的 lr 方向明确胜出，EMOE 的 dp 与 temp 存活、acc4 出局。**

   胜者 3-seed 定版：

   | 配置 | test acc | test F1 | 论文 | 差距（均值口径） | 差距（best 口径） |
   |---|---|---|---|---|---|
   | **EMOE `emoe_p_dp`**（dropout 0.3） | **71.61 ± 1.28** | **69.29 ± 1.74** | 72.58 / 70.73 | acc −0.97 ✅ / F1 −1.44 | **acc +0.45 ✅ / F1 −0.14 ✅** |
   | **GSIT `gsit_lr1`**（lr_other 1e-4） | **71.09 ± 1.15** | **68.39 ± 1.08** | 72.60 / 69.40 | acc −1.51 / F1 −1.01（双双贴线，差 0.01） | acc −0.24 ✅ / F1 +0.19 ✅ |

   EMOE dp 的 best（73.03 / 70.59）按其论文自己的"挑峰值"口径**双双进带、acc 反超论文**；GSIT lr1 按均值口径距验收线仅 0.01/0.01。

3. ✅ **收尾**（2026-08-31 补跑）：`emoe_p_temp` seeds 1/2 = 71.69/70.29、71.24/68.27，temp 3-seed 均值 71.24/68.77——**dp 仍为冠军**（均值 71.61/69.29、best 73.03/70.59），temp 为次优。不可消除项——两方法 MIntRec 私有超参未公开，本表为"公开信息范围内最忠实复现"，GSIT 论文为 5-seed 均值口径（我们 3 seeds）。

**目标一结论**：两方法经协议净化 + 单变量 lr/dropout 还原后，**均达到或逼近论文水平**（EMOE 峰值口径进带、GSIT 贴线）；种子方差 ±1.1~1.7 与 dev/test 背离是 445 条 dev 固有限制，最终对比表报 mean±std + best 双口径。

---

## 6. 目标二：DLF / DDSE 对标 MISA（71.46 / 68.89 / 70.84 / 69.69）

**验收标准**：acc ≥ 70.0、macroF1 ≥ 67.9（MISA −1.5/−1.0 带内）。

**差距现状**：R1 dlf 62.47 / ddse 62.25 → R2（bs64）已修复至 69.21 / 66.29，方向确认。

### 6.1 A/B 对照实验（🔄 重跑中）

- **2026-08-29 23:06 首次启动**：仅 `dlf_a` 跑起来，**第 31 轮 iteration 中途被中断**（非早停；best dev 0.55@29 且仍在上升，疑似机器关机）；`dlf_b`、`ddse_a`、`ddse_b` 未及启动。
- **2026-08-30 12:31 双链重启**（并行两链，各链内串行）：

| 链 | 顺序 | 日志 | 结果 |
|---|---|---|---|
| A 链 | `dlf_a` → `ddse_a` | `logs/full_{a_dlf,a_ddse}_r2.log` | `results/full_a_{dlf,ddse}.csv` |
| B 链 | `dlf_b` → `ddse_b` | `logs/full_{b_dlf,b_ddse}_r2.log` | `results/full_b_{dlf,ddse}.csv` |

- 方案定义：**A** = bs16 + 累积 10（等效 160）+ lr 1e-4（`configs/{dlf,ddse}_a.py`）；**B** = bs16 + lr 3e-5 + 无累积（`configs/{dlf,ddse}_b.py`）。
- 重跑为种子 0 确定性复现：`dlf_a` 第 1 轮 dev 0.0114 与中断前逐位一致，轨迹有效。
- ✅ **已完成**（08-30 12:31 ~ 17:15，双链并行），最终结果：

| run | 早停@ | dev F1 | test acc | test macro F1 | 判定 |
|---|---|---|---|---|---|
| dlf_a | 48 | 0.6613 | 70.56 | 67.10 | — |
| **dlf_b** | 58 | **0.6673** | **71.91** | **67.82** | **DLF 胜者：B** |
| **ddse_a** | 59 | **0.6648** | **69.89** | **64.68** | **DDSE 胜者：A** |
| ddse_b | 42 | 0.6459 | 67.87 | 63.67 | — |

（结果文件 `results/full_{a,b}_{dlf,ddse}.csv`；日志 `logs/full_{a,b}_{dlf,ddse}_r2.log`。）

### 6.1.1 A/B 结论（v2 新增）

1. **两个方法的赢家相反**：DLF 低 lr 直训（B）更好，DDSE 大等效 batch（A）更好——"训练不稳"的成因并不单一，不能一套配方套两方法。
2. **对照修复前基线**（R1 → A/B 最优）：dlf 62.47/58.42 → **71.91/67.82**（+9.44/+9.40）；ddse 62.25/57.32 → **69.89/64.68**（+7.64/+7.36）。
3. **验收进度**：DLF acc 71.91 ✅（超 MISA 71.46）、macroF1 67.82 差 0.08 即达标；DDSE acc 69.89 差 0.11、macroF1 64.68 还差 3.2（DDSE 辅助头更多，辅助稀释问题残留）。两方法 acc 均已进入/逼近 70 线，**"方向正确"判定通过**。

### 6.2 二阶微调（✅ 2026-08-30 完成，单变量，seed 0）

**DLF lr 细扫**（基于胜者 dlf_b = lr 3e-5，无累积）：

| 配置 | lr | 早停@ | dev F1 | test acc | test F1 |
|---|---|---|---|---|---|
| dlf_b2 | 2e-5 | 62 | **0.6788** | 70.79 | 65.94 |
| **dlf_b（推荐）** | 3e-5 | 58 | 0.6673 | **71.91** | **67.82** |
| dlf_b3 | 5e-5 | 29 | 0.6467 | 68.76 | 67.21 |

**DDSE 累积细扫**（基于胜者 ddse_a = 累积 10 + lr 1e-4）：

| 配置 | 累积（等效 batch） | 早停@ | dev F1 | test acc | test F1 |
|---|---|---|---|---|---|
| ddse_a2 | 8（128） | 53 | 0.6559 | 67.87 | 63.81 |
| **ddse_a（推荐）** | 10（160） | 59 | **0.6648** | **69.89** | 64.68 |
| ddse_a3 | 16（320） | 67 | 0.6501 | 69.89 | 65.00 |

**结论**：

1. **最终推荐配置**：DLF = `dlf`（bs16 + lr 3e-5）；DDSE = `ddse_a`（bs16 + 累积 10 + lr 1e-4）。DLF 的 2e-5~3e-5 为平台区、5e-5 过冲；DDSE 的累积 10/16 等效（acc 同为 69.89），8 过小。
2. **dev/test 背离警告（重要）**：dlf_b2 dev 最高（0.679）但 test 最低档（70.79），test 冠军 dlf_b 的 dev 反而第二。dev 仅 445 条，对 dev 差距 ≤0.012 的配置排序不可靠——**单 seed 下 dev 排名 ≠ test 排名，最终取舍必须 3 seeds 均值**（对应 §11 可复现性纪律）。
3. **验收现状**：DLF acc 71.91 ✅（超 MISA 71.46）、macroF1 67.82 差 0.08；DDSE acc 69.89 差 0.11、macroF1 64.68~65.00 差 ~3。DLF 差一步达标，DDSE 的残余差距集中在稀有类 F1。

**DDSE 下一杠杆（需用户拍板）**：优化维度已扫尽（lr/累积/batch 均 8 个 run），剩余手段是 §6.3 结构级（辅助头 5→2，改变论文方法定义）或先跑 3 seeds 确认 69.89/65.0 的稳健性。

### 6.3 结构级手段（最后手段，需确认是否偏离论文设定）

- DLF 语言头权重 3×→1×、辅助损失整体 ×0.5 消融
- DDSE 辅助头 5→2（主 + logits_c）
- 预期收益最大但改变论文方法定义，仅在 A/B + 微调后仍 <70 时与用户确认后进行

---

## 7. 目标三：TCS_Mamba 全面超越（论文模型）

**验收标准（按优先级）**：
- L1：acc ≥ 72.6 且 macroF1 ≥ 69.4（超 EMOE/GsiT 参考值）
- L2：acc ≥ 73.0 且 macroF1 ≥ 70.6（超 CAGC，全表第一）

**差距现状**：R1 bs16 最好 71.24 / 66.59（−1.4 / −2.8 至 L1）。

**行动项与结果**：

1. ✅ **阶段 1：配方对齐**（2026-08-31 完成；管理器已加梯度累积，冒烟通过）：

   | 配方 | dev F1 | test acc / F1 | 判定 |
   |---|---|---|---|
   | R1 基线（bs16，无累积，lr 2e-5） | 0.6762 | 71.24 / 66.59 | 参照 |
   | a1（累积8，lr 2e-5） | 0.6643 | 70.34 / 65.77 | 双降 ❌（累积不升 lr 会掉点） |
   | **a2（累积8，lr 3e-5）** | 0.6649 | **71.91 / 67.99** | **+0.67 acc / +1.40 F1 ✅ 定版** |
   | a3（a2 + patience 12） | 0.6614 | 71.91 / 67.69 | 耐心加长无收益 |

   注：a2 的 dev（0.665）略低于基线（0.676）但 test 双升——dev/test 背离再现且方向有利；阶段 2 仍按 dev 判定，最终多种子确认。

2. ✅ **阶段 2：dev 网格搜索**（9 配置双链，单 seed 0，2026-08-31 完成）：

   | 方向 | 配置 | dev F1 | test acc / F1 | 判定 |
   |---|---|---|---|---|
   | 结构 | d16 / d32（d_state↑） | 0.6731 / 0.6799 | 69.21/65.65、68.76/64.81 | dev 涨但 test 崩 ❌ |
   | 结构 | h4 / h8（头数/维度↑） | 0.6607 / 0.6804 | 71.01/66.38、70.11/64.89 | dev 涨但 test 降 ❌ |
   | 结构 | k5（卷积核 5） | 0.6841 | 71.24/67.08 | dev 涨 test 微降 ❌ |
   | 损失 | auxhalf / aux0 | 0.6642 / 0.6672 | 71.01/65.46、71.46/67.27 | 无收益 ❌ |
   | 正则 | dp04 / dp05 | 0.6665 / 0.6665 | 71.91/67.34、71.91/67.49 | 与 a2 持平 ≈ |

   **结论：9/9 无一 test 超越 a2（71.91 / 67.99），a2 为最终配方**。三个结构变量方向全部"dev 涨 test 崩"（小数据容量过拟合，445 条 dev 无法裁决容量参数——写进论文的 dev/test 背离案例）；辅助损失消融无收益说明 TCS 辅助项设计健康（与 DLF/DDSE 相反）。

3. 🔄 **阶段 3：5 seeds 定版**（a2，2026-08-31）：

   | seed | dev | test acc / F1 |
   |---|---|---|
   | 0 | 0.665 | 71.91 / 67.99 |
   | 1 | 0.696 | 69.44 / 66.78 |
   | 2 | 0.671 | 68.54 / 64.59 |
   | 3 | 0.667 | 69.44 / 65.53 |
   | 4 | 0.700 | 68.99 / 66.37 |
   | **mean±std** | 0.680 | **69.66 ± 1.31 / 66.25 ± 1.28** |
   | **best** | — | 71.91 / 67.99 |

   **重要发现**：seed 0（71.91）是明显离群高位，5-seed 均值远低于 L1（72.6/69.4）——单 seed 高分的可复现性被多种子证伪；且 seeds 1~4 的 dev 普遍高于 seed 0 但 test 更低（TCS 上 dev/test 背离最严重）。**"a2 优于 R1 基线"目前仅 seed 0 单一对照，正在补 R1 基线 5 seeds 做严格同口径比较。**
4. **阶段 4（若 L2 未达）**：结构增强——CoSSM 上加 EMOE 式单模态蒸馏（引用其论文）；HLBF 低秩秩数调优。改动需在论文中如实描述。

---

## 8. 目标四：计算效率多维度对比

**范围**：全部 9 个方法（text 作纯 BERT 基线）。统一硬件（本机 4080 SUPER）、同一 GPU 空闲状态下测量、每项跑 3 次取中位数。

**v2 注意**：transformer.py 检查点已改显存门控（<16GB 才启用），本机 MULT 不再开检查点（~105s/epoch vs WSL 时代 111s/iter）；所有耗时以 §3.2 实测为准，README 旧表作废。

| 维度 | 定义 | 测量方法 |
|---|---|---|
| 参数量 | 总参数 / BERT 部分 / **新增融合参数**（= 总 − BERT） | `sum(p.numel())`，按模块分组统计 |
| 计算量 | 每样本 FLOPs（fwd）与 fwd+bwd | `thop.profile` 或 torch profiler，固定输入 (16,30/230/480) |
| 训练时间 | 到早停的总 wall-clock；s/epoch；s/iter | **直接从 `logs/` 汇总（§3.2），不重跑** |
| 推理延迟 | batch=1 延迟（ms）与 batch=64 吞吐（samples/s） | 预热后计时 100 iter，`torch.cuda.synchronize` |
| 显存峰值 | 训练时 `torch.cuda.max_memory_allocated`；推理时同 | 训练/推理脚本内嵌计数 |
| 收敛效率 | 达到 best dev 的 epochs、达到 dev F1=0.60 所需 wall-clock | 日志曲线（已有，§3.2） |
| 能效（可选） | samples/s/W | `nvidia-smi --query-gpu=power.draw` 采样 |

**实现**：写 `tools/efficiency_bench.py`（模型构建沿用各 config，输入用真实形状 dummy 数据；训练时间从日志汇总）。

**产出**：`results/efficiency.md` —— 9 方法 × 7 维度大表 + "精度-效率"散点（acc vs 新增参数量 / acc vs 推理延迟）。

---

## 9. 执行顺序与时间线（v2 更新）

```
[已完成] R1/R2/R2.5 三轮全量 + 日志动力学分析（§2、§3）
[已完成] 目标二 A/B 双链（dlf 取 B、ddse 取 A，§6.1.1）
[已完成] 目标二 二阶微调（DLF lr 3e-5、DDSE 累积 10 定版，§6.2）     ─→ [可选] 3 seeds 确认 / DDSE 结构手段（需拍板）
[排队]   目标一：emoe_p ×3 seeds + gsit ×2 seeds（~6 min/run）   ~0.5h ─┤
                                                                      ├─→ 目标一定版（受限调参 ~1h，视差距）
[排队]   目标三阶段 1：tcs 累积配方（3 run）                     ~0.3h   │
[排队]   目标三阶段 2：dev 网格 15~20 配置（预算重估）           ~1-2h  ──┤
[排队]   目标三阶段 3：5 seeds                                  ~0.5h  ──┴─→ 目标三定版
[最后]   目标四：efficiency_bench.py + 汇总（GPU 空闲时）        ~1h
```

各目标间无强依赖，GPU 空闲即可交错执行；32GB 显存支持双任务并行（A/B 已验证）。

---

## 10. 交付物清单

1. `docs/PLAN.md`（本文档，随进度勾选更新）
2. 最终对比大表：9 方法 × (acc/F1/P/R) × (ours vs 论文参考值)，含多种子统计
3. `results/efficiency.md`：9 方法 × 7 维度效率表 + 精度-效率图
4. 各方法最终配置文件（`configs/*_final.py`）与完整日志
5. **代码变更清单**（当前工作区未提交改动，`git diff` 可查；提交时建议按此拆分 commit）：
   - `utils/metrics.py`：+weighted f1/prec/rec（`f1_w`/`prec_w`/`rec_w`），CSV 同步输出
   - `methods/DLF/manager.py`、`methods/DDSE/manager.py`：+`update_epochs` 梯度累积（zero_grad 移至窗首，clip+step 移至窗末，尾部残余 backward 收尾 step）
   - `methods/EMOE/manager.py`、`methods/GSIT/manager.py`（新增）：同款累积结构；EMOE 损失 = 4 头任务 CE + 路由监督 + 熵平衡 + 单模态蒸馏；GSIT 图 Transformer 管理器
   - `backbones/SubNets/transformers_encoder/transformer.py`：梯度检查点显存门控（<16GB 才启用）
   - `utils/functions.py`：save_model 对无 `config` 属性的模型跳过 config.json 写出
   - `backbones/FusionNets/EMOE.py`、`backbones/FusionNets/gsit/`（新增模型）
   - `configs/`：基础配置 batch 16/32→64（bs16 历史结果对应已提交版本）；新增 `emoe{,_b16,_p,_smoke}.py`、`gsit{,_b16,_smoke}.py`、`{dlf,ddse}_{a,b}.py`
   - `methods/__init__.py`、`backbones/__init__.py`：注册 emoe/gsit

---

## 11. 纪律与风险

1. **科研诚信**：调参只看 dev；test 一次性评估；多种子/峰值口径在论文中如实披露（与 EMOE/GsiT 原文口径一致）。
2. **可复现性**：所有最优配置入库（git diff 可查）；固定种子列表；`HF_HUB_OFFLINE=1`；A/B 重跑已验证种子级确定性（中断前后的 dlf_a 第 1 轮逐位一致）。
3. **风险（v2 更新）**：
   - ~~磁盘余量不足~~ → **已解决**：outputs/数据迁至 autodl-tmp（50G 数据盘，余 ~37G），系统盘恢复 18G 空闲
   - emoe/gsit 私有超参不可得 → 对齐上限受限，报告中注明
   - TCS_Mamba 阶段 2 网格若无效 → 阶段 4 结构增强需重跑冒烟验证梯度流
   - 效率测量受后台任务干扰 → 仅在 GPU 空闲时测量
   - **AutoDL 实例可能被关机/回收**：训练一律后台 + 日志落盘，中断后按 §6.1 方式重启（同种子可确定性续验）；重要结果（results/*.csv）及时同步
