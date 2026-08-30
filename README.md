# 多模态意图识别(MIntRec)九方法统一对比框架

本仓库在 [MIntRec](https://github.com/thuiar/MIntRec)(ACM MM 2022)官方框架基础上,迁移并统一实现了 **9 个多模态方法**,在统一的数据加载、训练协议与评估流程下进行多模态意图识别(20 类)对比。原英文说明见 [README_en.md](./README_en.md);本地调试排查记录见 [ENV_NOTES.md](./ENV_NOTES.md);实验计划与进度见 [docs/PLAN.md](./docs/PLAN.md)。

| 方法 | 来源 | 架构一句话 | Mamba 内核 |
|---|---|---|---|
| `text` | MIntRec 原生 | 纯文本 BERT 分类基线 | 不需要 |
| `misa` | MIntRec 原生 | [MISA](https://github.com/declare-lab/MISA) 模态不变/专属子空间 | 不需要 |
| `mult` | MIntRec 原生 | [MULT](https://github.com/yaohungt/Multimodal-Transformer) 跨模态 Transformer | 不需要 |
| `mag_bert` | MIntRec 原生 | [MAG-BERT](https://github.com/WasifurRahman/BERT_multimodal_transformer) 模态感知门控 | 不需要 |
| `tcs_mamba` | 迁移 | Tucker 共享子空间 + CP 私有子空间 + CoSSM 双向跨模态 Mamba + HLBF 融合 | **需要** |
| `dlf` | [DLF, AAAI 2025](https://github.com/pwang322/DLF) | 语言焦点解耦,LFA 跨模态注意力,8 个预测头 | 不需要 |
| `ddse` | [DDSE, ACM MM 2025](https://github.com/jiangshenjie/DDSE) | 文本中心 SSM(TCSSM + TSSA)双流解耦增强 | **需要** |
| `emoe` | EMOE, CVPR 2025 | 模态专家混合:路由器加权 + 单模态蒸馏 + 熵平衡 | 不需要 |
| `gsit` | GsiT | 图结构跨模态 Transformer(CMG + 图注意力) | 不需要 |

所有方法共用同一套训练循环协议:AdamW + ReduceLROnPlateau、EarlyStopping(patience 8,监控 **dev macro F1**,保存最优权重副本用于测试)、梯度累积可选(`update_epochs`)。

---

## 1. Python 环境总览(重要)

**实测通过的完整版本组合**(两台机器验证,请尽量对齐大版本):

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | **3.12.3** | conda 创建 |
| PyTorch | **2.8.0+cu128** | torchvision 0.23.0+cu128 |
| CUDA(驱动) | 13.0(驱动 580.105.08) | cu128 轮子向下兼容新驱动 |
| transformers | **5.5.4** | ⚠️ 4.x 老接口不兼容,详见第 10 节 |
| pandas | **3.0.2** | 2.x 及以下无 `pd.concat` 之外的差异即可 |
| numpy | 2.3.2 | |
| scikit-learn | 1.8.0 | 指标计算 |
| mamba-ssm | **2.3.1** | 仅 `tcs_mamba` / `ddse` 需要 |
| causal-conv1d | **1.7.0** | 同上 |
| triton | 3.4.0 | 随 torch 安装 |
| ninja | 1.13.0 | mamba 编译加速,强烈建议 |
| easydict / einops / tqdm / matplotlib | 1.13+ / 0.8.2 / 4.66.2 / 3.10.5 | 基础依赖 |

**实测 GPU**:RTX 4080 SUPER 32GB(Ada,sm_89,原生 Linux)与 RTX 5060 Laptop 8GB(Blackwell,sm_120,WSL2);A100/3090/4090 等常见云 GPU 均可。显存 ≥ 8GB 即可跑全部方法(9 方法单 run 峰值 < 4GB;32GB 显存可双任务并行)。

**注意**:`torch` / `torchvision` / `mamba-ssm` / `causal-conv1d` 不在 `requirements.txt` 里(前者需按 CUDA 环境选择安装源,后者需编译环境,见第 3、5 节),其余依赖一键安装。

## 2. 创建 conda 环境

```bash
conda create -n mintrec python=3.12 -y
conda activate mintrec
```

## 3. 安装 PyTorch(按 GPU 的 CUDA 环境选择)

先确认驱动支持的 CUDA 版本:`nvidia-smi` 右上角显示的 CUDA Version。

```bash
# 新驱动(CUDA 12.8+,覆盖 Blackwell/Ada/Ampere,推荐,与实测环境一致)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 老驱动(CUDA 11.8,如部分 A10/3090 环境)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

验证:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch; print(torch.version.cuda)"   # 记住这个版本,第 5 节装 nvcc 要对齐
```

## 4. 安装其余依赖

```bash
pip install -r requirements.txt
```

## 5. 安装 Mamba CUDA 内核(mamba-ssm / causal-conv1d)

**只有 `tcs_mamba` 和 `ddse` 两个方法需要**;只跑其余 7 个方法可整节跳过。

### 5.1 原理与前置条件

这两个包首次安装时会**现场编译 CUDA 扩展**(`selective_scan_cuda` / `causal_conv1d_cuda`),因此要求:

1. **环境里有 nvcc,且大版本与 torch 的 CUDA 一致**(例如 torch 为 `2.8.0+cu128`,则 nvcc 需为 12.8.x)。查 torch 侧版本:`python -c "import torch; print(torch.version.cuda)"`;查 nvcc:`nvcc --version`。
2. nvcc 推荐用 conda 装(不污染系统):

   ```bash
   conda install -c nvidia/label/cuda-12.8.1 cuda-toolkit -y   # 版本号按上一步对齐
   ```

3. `pip install ninja`(并行编译,缺了会慢很多也能编)。
4. 编译需要数 GB 空闲内存 + 每个 包几分钟时间,属正常现象。

### 5.2 安装

```bash
# 先 causal-conv1d,后 mamba-ssm(mamba 会链接已装的 causal_conv1d)
pip install causal-conv1d>=1.6
pip install mamba-ssm>=2.3
# 或者直接: pip install -r requirements.txt(内含这两行,顺序正确)
```

pip 会优先下载与 torch/cuda/python 匹配的**官方预编译 wheel**(几秒钟);匹配不到才回落源码编译。与安装相关的环境变量:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"    # 只编译本机架构,大幅加速:4080S/4090 用 8.9,50 系用 12.0,A100 用 8.0
export MAX_JOBS=4                    # 编译并发,内存紧张时调小
# 极少用:强制跳过 wheel 直接源码编译
# MAMBA_FORCE_BUILD=TRUE pip install mamba-ssm --no-build-isolation
# CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install causal-conv1d --no-build-isolation
```

`--no-build-isolation` 用于解决"构建隔离环境里找不到 nvcc/torch"的问题。

### 5.3 验证

```bash
python -c "import mamba_ssm, causal_conv1d, selective_scan_cuda, causal_conv1d_cuda; print('mamba kernels OK')"
```

### 5.4 常见问题

| 现象 | 原因与处理 |
|---|---|
| `nvcc not found` / CMake 报找不到 CUDA | nvcc 未装或不在 PATH;`conda install -c nvidia/label/cuda-<ver> cuda-toolkit` 后重开 shell |
| 编译报 CUDA 版本与 torch 不匹配 | `nvcc --version` 与 `torch.version.cuda` 大版本必须一致 |
| `No kernel image is available` / 不支持的架构 | 包版本太老不认识新卡;升级 `mamba-ssm`/`causal-conv1d` 最新版,并设 `TORCH_CUDA_ARCH_LIST` |
| import 时 `undefined symbol` | **torch 升级后必须重装这两个包**(扩展二进制绑定 torch 版本) |
| 编译过程 OOM / 卡死 | `export MAX_JOBS=2` 降并发后重试 |

## 6. 准备 MIntRec 数据集

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

## 7. 准备 bert-base-uncased 预训练权重

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

之后训练可全程离线:`export HF_HUB_OFFLINE=1`。

## 8. 运行

统一命令模板(在仓库根目录执行):

```bash
export HF_HUB_OFFLINE=1
PY=python

$PY run.py --dataset MIntRec --method <method> --data_mode multi-class \
  --train --save_results --save_model --seed 0 --gpu_id 0 \
  --text_backbone bert-base-uncased \
  --config_file_name <config> --results_file_name <method>.csv --num_workers 0
```

`<method>` / `<config>` 对照:

| --method | --config_file_name | 备注 |
|---|---|---|
| `text` | `text_bert` | |
| `misa` | `misa_bert` | |
| `mult` | `mult_bert` | 最慢;≥16GB 显存的 GPU 上梯度检查点自动关闭(更快) |
| `mag_bert` | `mag_bert` | |
| `tcs_mamba` | `tcs_mamba` | 需 mamba 内核 |
| `dlf` | `dlf` | |
| `ddse` | `ddse` | 需 mamba 内核 |
| `emoe` | `emoe`(bs64+累积2)或 `emoe_b16`(bs16+累积10) | |
| `gsit` | `gsit`(bs64)或 `gsit_b16`(bs16+累积4) | |

`configs/` 里还有三类变体:`*_smoke.py`(2 epochs 冒烟自检)、`*_bi.py`(binary-class 模式)、`dlf_{a,b}.py` / `ddse_{a,b}.py` / `emoe_p.py`(研究实验变体,含义见 docs/PLAN.md)。

其他常用参数:

- `--data_mode binary-class`:二分类模式(Emotion/Goal),20 类细粒度标签会被映射
- `--seed`:随机种子;`--results_file_name`:结果追加写入的 CSV 文件名
- 结果:`results/<name>.csv`(acc / macro F1 / P / R,×100,另附 weighted 列作参考);日志:`logs/`;最优模型:`outputs/`
- `--num_workers`:0 最保守(任何机器都能跑);大内存机器可用 2~4 提速,不影响结果

## 9. 实测训练时长(RTX 4080 SUPER 32GB,单 seed 全程含早停)

| 方法 | 协议 | 到早停总耗时 | 最优轮次 |
|---|---|---|---|
| text | bs16 | ~2 分钟 | 23/30 |
| misa | bs16 | ~2 分钟 | 5/12 |
| tcs_mamba | bs16 | ~2.6 分钟 | 11/18 |
| mag_bert | bs16 | ~3.7 分钟 | 21/28 |
| emoe | bs16+累积10 | ~6 分钟 | 7/14 |
| gsit | bs16+累积4 | ~7 分钟 | 10/17 |
| ddse | bs32 | ~15 分钟 | 23/30 |
| dlf | bs16 | ~20 分钟 | 12/19 |
| mult | bs16 | ~23 分钟 | 6/13 |

要点:所有方法都带 EarlyStopping(patience 8,监控 dev macro F1),**100 epochs 只是上限,实测从不跑满**(最优轮 5~31 即收敛),下表是真实成本;九方法全跑约 80 分钟。RTX 5060 Laptop 8GB 上约慢 2~4 倍,且必须 `--num_workers 0`(小内存 OOM 风险,见 FAQ)。

## 10. 新版依赖兼容性说明

原 MIntRec 锁定 transformers 4.17 + pandas 1.1.5 + Python≤3.9,无法在 Blackwell GPU 上运行,本仓库已整体适配到现代依赖(见第 1 节版本表):

| 问题 | 修复 |
|---|---|
| transformers 5.x 移除 `AdamW` | 改用 `torch.optim.AdamW` |
| transformers 5.x 自定义 `PreTrainedModel` 子类必须调 `post_init()` | MAG_BERT / BERTEncoder 等已改 |
| transformers 5.x 移除 `get_head_mask`、`get_extended_attention_mask` 第三参数改为 dtype | MAG_BERT 内联实现 |
| transformers 5.x meta-device 加载使检查点外的 `nn.LSTM` 权重未初始化(曾导致 mag_bert NaN) | 加载后对 `nn.RNNBase` 显式 `reset_parameters()` |
| pandas 3.0 移除 `DataFrame.append` | 改 `pd.concat` |
| torch 2.2+ 移除 `ReduceLROnPlateau(verbose=)` | 去参 |
| 音视频特征 float64 padding 导致内存爆炸 | 特征转 float32,张量构建后逐 split 释放 |
| MULT 激活值超小显存(8GB 卡 30s/iter) | TransformerEncoder 加梯度检查点,**并按显存门控**(≥16GB 自动关闭) |
| DLF / DDSE / EMOE / GSIT / TCS_Mamba 管理器 | 统一支持 `update_epochs` 梯度累积 |
| 部分自定义模型无 `config` 属性导致 save_model 崩溃 | `save_model` 对无 config 的模型跳过 config.json 写出 |
| 评估口径 | Metrics 增加 weighted F1/P/R 列(CSV 参考列),主表仍为 macro |
| DDSE 的 TCSSM 要求双流等长,而 MIntRec 三模态 26/476/476 不等长 | a/v 投影后 `F.interpolate` 到文本长度(等价论文 aligned 模式) |

迁移要点与训练损失构成见 [ENV_NOTES.md](./ENV_NOTES.md);原版依赖环境保留在 `requirements_legacy.txt`(仅 `tools/`、`TalkNet_ASD/` 特征抽取需要,训练不用)。

## 11. 目录结构

```
├── run.py                  # 唯一入口:--method + --config_file_name 驱动
├── configs/                # 每方法一个超参配置(变体见第 8 节)
├── data/                   # 三模态预处理与 DataManager(tsv + pkl 特征加载)
├── backbones/
│   ├── SubNets/            # BERTEncoder、fairseq 式 TransformerEncoder
│   └── FusionNets/         # 9 个方法的融合模型(tcs_mamba/、ddse/、gsit/ 为子包)
├── methods/                # 每方法一个 manager.py(训练循环、损失、早停)
├── utils/                  # EarlyStopping、Metrics(macro+weighted)、hinge_loss
├── tools/                  # 原始音视频特征抽取(需 legacy 依赖)
├── MIA-datasets/           # 数据集(gitignore,见第 6 节)
├── cache/                  # bert-base-uncased 本地缓存(gitignore)
├── logs/  outputs/  results/   # 日志 / checkpoint / 结果 CSV(均 gitignore)
└── docs/PLAN.md            # 实验计划与进度
```

## 12. 常见问题(FAQ)

- **OOM(EXIT 137,进程无声消失)**:小内存机器务必 `--num_workers 0`;仍 OOM 则调小 `configs/<method>.py` 里 `train_batch_size`
- **GPU 显存 OOM**:调小 `train_batch_size` / `eval_batch_size`;mult 在小显存卡上自动启用梯度检查点
- **`Can't load the configuration of 'bert-base-uncased'`**:网络不通,设置 `export HF_ENDPOINT=https://hf-mirror.com` 或用本地权重路径(第 7 节方式 B)
- **`mamba-ssm` 安装失败**:逐条对照第 5.4 节;不跑 `tcs_mamba`/`ddse` 可完全跳过
- **torch 升级后 `tcs_mamba`/`ddse` import 报 `undefined symbol`**:重装 `mamba-ssm` 与 `causal-conv1d`(见第 5.4 节)
- **`KeyError: 'xxx_seq_len'` / 维度不匹配**:确认数据目录结构正确、软链接已建(第 6 节)
- **磁盘被 outputs 撑满**:checkpoint 增长快,可整目录迁到大容量数据盘后软链接接回(本仓库迁移实例见 docs/PLAN.md §1.2)
- **评估指标**:主口径为 macro F1/P/R(与 MIntRec 官方一致),CSV 同时保留 weighted 列;20 类随机基线 acc ≈ 5%

## 13. 引用

使用本仓库或数据集请引用 MIntRec 原论文;使用了 `dlf` / `ddse` / `emoe` / `gsit` / `tcs_mamba` 对应方法请同时引用其原论文(DLF: AAAI 2025;DDSE: ACM MM 2025;EMOE: CVPR 2025;GsiT 见其官方仓库)。

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

## 14. 致谢

代码源自 [MIntRec](https://github.com/thuiar/MIntRec)、[MMSA](https://github.com/thuiar/MMSA)、[DLF](https://github.com/pwang322/DLF)、[DDSE](https://github.com/jiangshenjie/DDSE)、[MULT](https://github.com/yaohungt/Multimodal-Transformer)、[MISA](https://github.com/declare-lab/MISA)、[MAG-BERT](https://github.com/WasifurRahman/BERT_multimodal_transformer)、EMOE、GsiT 等开源项目,感谢原作者。
