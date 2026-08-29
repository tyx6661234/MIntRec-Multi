# 多模态意图识别(MIntRec)七方法对比框架

本仓库在 [MIntRec](https://github.com/thuiar/MIntRec)(ACM MM 2022)官方框架基础上,迁移了 **TCS_Mamba**、**DLF**、**DDSE** 三个多模态方法,共 **7 个方法**在统一的数据加载、训练协议与评估流程下进行多模态意图识别(20 类)对比。原英文说明见 [README_en.md](./README_en.md);本地调试排查记录见 [ENV_NOTES.md](./ENV_NOTES.md)。

| 方法 | 来源 | 架构一句话 |
|---|---|---|
| `text` | MIntRec 原生 | 纯文本 BERT 分类基线 |
| `misa` | MIntRec 原生 | [MISA](https://github.com/declare-lab/MISA) 模态不变/专属子空间 |
| `mult` | MIntRec 原生 | [MULT](https://github.com/yaohungt/Multimodal-Transformer) 跨模态 Transformer |
| `mag_bert` | MIntRec 原生 | [MAG-BERT](https://github.com/WasifurRahman/BERT_multimodal_transformer) 模态感知门控 |
| `tcs_mamba` | 迁移 | Tucker 共享子空间 + CP 私有子空间 + CoSSM 双向跨模态 Mamba + HLBF 融合 |
| `dlf` | [DLF, AAAI 2025](https://github.com/pwang322/DLF) | 语言焦点解耦,LFA 跨模态注意力,8 个预测头 |
| `ddse` | [DDSE, ACM MM 2025](https://github.com/jiangshenjie/DDSE) | 文本中心 SSM(TCSSM + TSSA)双流解耦增强 |

## 1. 硬件与系统要求

- **系统**:Linux(原生或 WSL2 均可)
- **GPU**:显存 ≥ 8GB 的 NVIDIA GPU。开发环境为 RTX 5060 Laptop(Blackwell,sm_120);A100/3090/4090 等常见云 GPU 均可
- **磁盘**:约 10GB(环境 ~8GB + 数据 ~2GB)
- **内存**:≥ 16GB 推荐(开发机 7.4GB 已可跑,但必须遵守第 6 节的 `--num_workers 0` 约束)

## 2. 创建 conda 环境

```bash
conda create -n mintrec python=3.12 -y
conda activate mintrec
```

## 3. 安装 PyTorch(按 GPU 的 CUDA 环境选择)

先确认驱动支持的 CUDA 版本:`nvidia-smi` 右上角显示的 CUDA Version。

```bash
# 新驱动(CUDA 12.8+,覆盖 Blackwell/Ada/Ampere,推荐)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 老驱动(CUDA 11.8,如部分 A10/3090 环境)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

验证:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

**关于 Mamba CUDA 内核(`mamba-ssm` / `causal-conv1d`)**:

- 仅 `tcs_mamba` 和 `ddse` 两个方法需要;只跑其余五个方法可跳过
- 这两个包首次安装时会**现场编译 CUDA 扩展**,要求环境里有与 torch CUDA 版本一致的 **nvcc**(如 `conda install -c nvidia cuda-toolkit=12.8` 或系统 CUDA toolkit),且编译需数分钟
- 若编译失败,先确认 `nvcc --version` 可用且大版本与 `python -c "import torch; print(torch.version.cuda)"` 一致

验证(装了 mamba 内核的情况下):

```bash
python -c "import mamba_ssm, causal_conv1d, selective_scan_cuda, causal_conv1d_cuda; print('mamba kernels OK')"
```

## 5. 准备 MIntRec 数据集

从官方渠道下载 MIA-datasets:[Google Drive](https://drive.google.com/drive/folders/18iLqmUYDDOwIiiRbgwLpzw76BD62PK0p?usp=sharing) 或 [百度网盘](https://pan.baidu.com/s/1xWcrNL-lUiUSLklnozyQvQ)(提取码 **95lo**)。

只需其中:`train.tsv`、`dev.tsv`、`test.tsv`、`audio_data/audio_feats.pkl`、`video_data/video_feats.pkl`(约 2GB;`raw_data`/`speaker_annotations` 训练不需要)。

放到仓库根的 `MIA-datasets/` 下,并建立代码期望的目录结构(代码会拼上数据集名 `MIntRec`,用软链接即可,不必移动文件):

```bash
cd MIA-datasets
ln -s "$(pwd)" MIntRec
cd ..
# 校验:以下三个路径都应存在
ls MIA-datasets/MIntRec/train.tsv MIA-datasets/MIntRec/audio_data/audio_feats.pkl MIA-datasets/MIntRec/video_data/video_feats.pkl
```

## 6. 准备 bert-base-uncased 预训练权重

二选一:

```bash
# 方式 A:从 HuggingFace 镜像下载到本地缓存(推荐,离线可复用)
export HF_ENDPOINT=https://hf-mirror.com
python -c "from transformers import BertModel, BertTokenizer; \
  BertModel.from_pretrained('bert-base-uncased', cache_dir='cache'); \
  BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='cache')"

# 方式 B:已有本地完整权重目录(config.json + pytorch_model.bin + vocab.txt 等),直接用路径
#   运行时传 --text_backbone /path/to/bert_base_uncased 即可
```

## 7. 运行

统一命令模板(在仓库根目录执行):

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 使用方式 A 的缓存时不再联网
PY=python

$PY run.py --dataset MIntRec --method <method> --data_mode multi-class \
  --train --save_results --save_model --seed 0 --gpu_id 0 \
  --text_backbone bert-base-uncased \
  --config_file_name <config> --results_file_name <method>.csv \
  --num_workers 0
```

`<method>` / `<config>` 对照:

| --method | --config_file_name | 备注 |
|---|---|---|
| `text` | `text_bert` | |
| `misa` | `misa_bert` | |
| `mult` | `mult_bert` | 最慢,约 3 小时/全量 |
| `mag_bert` | `mag_bert` | |
| `tcs_mamba` | `tcs_mamba` | 需 mamba 内核 |
| `dlf` | `dlf` | |
| `ddse` | `ddse` | 需 mamba 内核,batch 32 |

**`--num_workers 0` 是硬约束**:num_workers ≥ 2 会在小内存机器上触发 dataloader 子进程 OOM(开发机上实测 EXIT 137);`dlf_smoke` 等冒烟配置同理。

其他常用参数:

- `--data_mode binary-class`:二分类模式(Emotion/Goal),20 类细粒度标签会被映射
- `--seed`:随机种子;`--results_file_name`:结果追加写入的 CSV 文件名
- 结果:`results/<name>.csv`(acc / f1 / prec / rec,×100);日志:`logs/`;最优模型:`outputs/`

### 全量训练时长参考(RTX 5060 Laptop 8GB 实测外推)

| 方法 | 单 epoch | 100 epochs 上限 | 早停后预估 |
|---|---|---|---|
| text | ~10s | ~17 分钟 | ~10–15 分钟 |
| mag_bert | ~14s | ~24 分钟 | ~15–20 分钟 |
| tcs_mamba | ~14s | ~24 分钟 | ~15–20 分钟 |
| misa | ~20s | ~33 分钟 | ~20–30 分钟 |
| ddse | ~58s | ~1.6 小时 | ~1–1.3 小时 |
| dlf | ~101s | ~2.8 小时 | ~1.5–2.5 小时 |
| mult | ~111s | ~3.1 小时 | ~2–3 小时 |
| **七方法合计** | | **≈ 9.5 小时** | **≈ 5–7 小时** |

云 GPU(A100/4090)通常更快。所有方法均带 EarlyStopping(patience 8,监控 dev F1),很少跑满 100 epochs。

### 建议执行顺序

轻量方法先出结果,耗时方法压后:

```
text → mag_bert → tcs_mamba → misa → ddse → dlf → mult
```

## 8. 新版依赖兼容性说明

原 MIntRec 锁定 transformers 4.17 + pandas 1.1.5 + Python≤3.9,无法在 Blackwell GPU 上运行,本仓库已整体适配到现代依赖(实测组合:Python 3.12 / torch 2.11+cu128 / transformers 5.5 / pandas 3.0):

| 问题 | 修复 |
|---|---|
| transformers 5.x 移除 `AdamW` | 改用 `torch.optim.AdamW` |
| transformers 5.x 自定义 `PreTrainedModel` 子类必须调 `post_init()` | MAG_BERT / BERTEncoder 等已改 |
| transformers 5.x 移除 `get_head_mask`、`get_extended_attention_mask` 第三参数改为 dtype | MAG_BERT 内联实现 |
| transformers 5.x meta-device 加载使检查点外的 `nn.LSTM` 权重未初始化(曾导致 mag_bert NaN) | 加载后对 `nn.RNNBase` 显式 `reset_parameters()` |
| pandas 3.0 移除 `DataFrame.append` | 改 `pd.concat` |
| torch 2.2+ 移除 `ReduceLROnPlateau(verbose=)` | 去参 |
| 音视频特征 float64 padding 导致内存爆炸 | 特征转 float32,张量构建后逐 split 释放 |
| MULT 激活值超 8GB 显存(30s/iter) | TransformerEncoder 加梯度检查点(1.1s/iter) |
| DDSE 的 TCSSM 要求双流等长,而 MIntRec 三模态 26/476/476 不等长 | a/v 投影后 `F.interpolate` 到文本长度(等价论文 aligned 模式) |

迁移要点与训练损失构成见 [ENV_NOTES.md](./ENV_NOTES.md);原版依赖环境保留在 `requirements_legacy.txt`(仅 `tools/` 特征抽取需要,训练不用)。

## 9. 常见问题(FAQ)

- **OOM(EXIT 137,进程无声消失)**:确认用了 `--num_workers 0`;仍 OOM 则调小 `configs/<method>.py` 里 `train_batch_size`
- **GPU 显存 OOM**:调小 `train_batch_size` / `eval_batch_size`;mult 已内置梯度检查点
- **`Can't load the configuration of 'bert-base-uncased'`**:网络不通,设置 `export HF_ENDPOINT=https://hf-mirror.com` 或用本地权重路径
- **`mamba-ssm` 安装失败**:见第 4 节,检查 nvcc 与 torch CUDA 版本一致性;不跑 mamba 系方法可跳过
- **`KeyError: 'xxx_seq_len'` / 维度不匹配**:确认数据目录结构正确、软链接已建
- **评估指标**:20 类随机基线 acc ≈ 5%;七方法 2-epoch 冒烟值见 [ENV_NOTES.md](./ENV_NOTES.md)

## 10. 引用

使用本仓库或数据集请引用 MIntRec 原论文;使用了 `dlf` / `ddse` / `tcs_mamba` 对应方法请同时引用其原论文(DLF: AAAI 2025;DDSE: ACM MM 2025)。

```bibtex
@inproceedings{10.1145/3503161.3547906,
   author = {Zhang, Hanlei and Xu, Hua and Wang, Xin and Zhou, Qianrui and Zhao, Shaojie and Teng, Jiayan},
   title = {MIntRec: A New Dataset for Multimodal Intent Recognition},
   year = {2022},
   doi = {10.1145/3503161.3547906},
   booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
   pages = {1688–1697},
   numpages = {10}
}
```

## 11. 致谢

代码源自 [MIntRec](https://github.com/thuiar/MIntRec)、[MMSA](https://github.com/thuiar/MMSA)、[DLF](https://github.com/pwang322/DLF)、[DDSE](https://github.com/jiangshenjie/DDSE)、[MULT](https://github.com/yaohungt/Multimodal-Transformer)、[MISA](https://github.com/declare-lab/MISA)、[MAG-BERT](https://github.com/WasifurRahman/BERT_multimodal_transformer) 等开源项目,感谢原作者。
