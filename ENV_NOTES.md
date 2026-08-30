# MIntRec 运行环境备注(WSL)

> **状态说明(2026-08-30)**:本文件是 2026-08-29 在 WSL(RTX 5060 Laptop 8GB)上的调试与迁移记录,保留作历史排查参考。
> 现行安装文档与精确版本表以 [README.md](./README.md) 为准;2026-08-30 起主开发环境已迁移至 AutoDL(RTX 4080 SUPER 32GB,`outputs/` 与 `MIA-datasets/` 迁至数据盘 `/root/autodl-tmp`),下文涉及的 `--num_workers 0` 硬约束、`/mnt/d/...` 路径等仅对旧 WSL 环境有效。

## Python 环境
- conda 环境:`mamba`(`/home/tyx/miniconda3/envs/mamba`)
- 解释器:`/home/tyx/miniconda3/envs/mamba/bin/python`
- 版本:Python 3.12.13 / torch 2.11.0+cu128 / transformers 5.5.4 / pandas 3.0.2 / scikit-learn 1.8.0
- GPU:RTX 5060 Laptop(Blackwell,需 torch>=2.7+cu128,老版 torch 不可用)

## 本地 bert-base-uncased 权重(2026-08-29 确认可用)
- Windows 路径:`D:\MMSA\bert_base_uncased`
- WSL 路径:`/mnt/d/MMSA/bert_base_uncased`
- 内容:config.json / pytorch_model.bin / tokenizer.json / tokenizer_config.json / vocab.txt
- 验证:BertModel + BertTokenizer 在 mamba 环境加载正常(109M 参数)
- 备用:网络镜像 `HF_ENDPOINT=https://hf-mirror.com`,`MIntRec/cache/` 里也已缓存一份

## 用法
把 `--text_backbone` 直接指向本地路径即可,例如:
```bash
PY=/home/tyx/miniconda3/envs/mamba/bin/python
$PY run.py --dataset MIntRec --method text --data_mode multi-class --train --save_results \
  --seed 0 --gpu_id 0 --text_backbone /mnt/d/MMSA/bert_base_uncased \
  --config_file_name text_bert --results_file_name text.csv --num_workers 2
```
注意:`data/text_pre.py` 的 tokenizer 仍固定加载 'bert-base-uncased',会命中
`MIntRec/cache/` 本地缓存,无需联网;如需彻底离线可加 `HF_HUB_OFFLINE=1`。

## 数据集(2026-08-29 就位)
- 位置:`MIntRec/MIA-datasets/`(train 1334 / dev 445 / test 445)
- 代码期望 `MIA-datasets/MIntRec/...`,用软链接 `MIA-datasets/MIntRec -> MIA-datasets` 对齐,数据文件未移动

## 本次适配新增的关键修复(冒烟验证通过)

| 问题 | 根因 | 修复 |
|---|---|---|
| 特征加载后进程静默死亡 | WSL 仅 7.4GB 内存;audio padding 到 480 帧 float64 ≈ 4GB,堆叠再翻倍 OOM | audio/video padding 输出 float32;`MMDataset` 用 `np.stack`+`from_numpy`;`data/base.py` 逐 split 构建后立即释放特征列表 |
| mult 训练 30s/iter | 激活值峰值 8.33GiB > 8GB 显存,驱动向内存倒灌 | `transformers_encoder/transformer.py` 层循环加梯度检查点(`torch.utils.checkpoint`,训练时启用)→ 1.1s/iter、峰值 2.95GiB |
| mag_bert acc 12.8% / loss=NaN | transformers 5.x `from_pretrained` 在 meta 设备上构建模型,检查点中不存在的 `nn.LSTM`(AlignSubNet CTC)丢失构造期初始化,`_init_weights` 又不认识 RNN,留下未初始化内存(含 NaN 字节) | `MAG_BERT.py` wrapper 加载后对所有 `nn.RNNBase` 模块调用 `reset_parameters()` |
| CPU 前向崩溃 | MAG.forward 里 `torch.ones(...).to(args.device)` 写死 GPU | 改为 `device=hm_norm.device` |
| 加载报错 all_tied_weights_keys / get_head_mask / mask 签名 | transformers 5.x 移除/变更 | 自定义子类改调 `post_init()`;`get_head_mask` 内联化;`get_extended_attention_mask` 去掉 device 实参 |

## 冒烟结果(2 epochs,multi-class,seed 0)
- text: acc 43.6 ✅  misa: acc 68.8 ✅  mult: acc 70.6 ✅  mag_bert: acc 45.8 ✅(修复后)
- 结果文件:`results/smoke_*.csv`;全量配置 `configs/*_smoke.py`(epochs=2)仅冒烟用

## 全量训练(2026-08-29 启动,后台)
- 命令模板:`run.py --dataset MIntRec --method {text|misa|mult|mag_bert} --data_mode multi-class --train --save_results --save_model --seed 0 --gpu_id 0 --text_backbone bert-base-uncased --config_file_name {text_bert|misa_bert|mult_bert|mag_bert} --results_file_name full_*.csv --num_workers 2`
- 日志:`/tmp/full_{text,misa,mult,mag_bert}.log`;结果:`results/full_*.csv`
- 论文参考值(multi-class):text 44.3 / mult 73.9 / misa 72.8 / mag_bert 75.0(accuracy)

## TCS_Mamba 迁移(2026-08-29 完成,冒烟通过)
- 接入为第五个方法 `tcs_mamba`,与 text/misa/mult/mag_bert 同一套 MIntRec 训练协议
- 新增文件:
  - `backbones/FusionNets/TCS_Mamba.py`(主模型适配版:Tucker+CP+CoSSM+HLBF,forward(text,video,audio)→logits,辅助量存 `model.aux_outputs`)
  - `backbones/FusionNets/tcs_mamba/`(cosmoss.py 从 DepMamba.py 剥离,speechbrain→nn.SiLU/LayerNorm;mamba/mm_bimamba.py+selective_scan_interface.py 原样迁移)
  - `methods/TCS_Mamba/manager.py`(MISA 模板 + 辅助损失:CE + 0.1×CE(shared/private) + λ_rec×recon + λ_ort×orth)
  - `configs/tcs_mamba.py` / `configs/tcs_mamba_smoke.py`
- 修改注册:`methods/__init__.py`、`backbones/__init__.py`
- 适配点:`feature_dims` 兼容(DataManager 只注入单个 feat_dim 字段)
- 梯度流已验证;GatedCoSSM 设计上丢弃 cossm 文本侧输出,故 a_in_proj 无梯度属正常
- **注意:此方法需 `--num_workers 0`**(WSL 7.4GB 内存紧张,2 个 dataloader worker 会挤爆内存,EXIT 137)
- 冒烟(2 epochs):acc 61.57 / f1 50.35,eval F1 0.33→0.50 上升正常
- CUDA 核依赖:mamba_ssm 2.3.1 + causal_conv1d 1.6.1(已装),selective_scan_cuda/causal_conv1d_cuda 可用

## DLF / DDSE 迁移(2026-08-29 完成,冒烟通过)
- **DLF**(AAAI 2025,源:/mnt/d/TCS_Mamba/DLF/):`backbones/FusionNets/DLF.py` + `methods/DLF/manager.py` + `configs/dlf.py`;语言焦点解耦,8 头(主输出/c/3 hetero/3 low,全部 num_labels),损失 = 任务 CE(语言头 3×)+ 重构 + 共享一致性 + 正交(reshape(-1, d))+ Hinge,组合式 task + (s_sr+recon+(sim+ort)*0.1)*0.1
- **DDSE**(ACM MM25,源:/mnt/d/TCS_Mamba/DDSE/):`backbones/FusionNets/DDSE.py` + `backbones/FusionNets/ddse/`(TSSA.py 原样;TCSSM 直接复用 tcs_mamba 的 CoSSM 别名,结构相同)+ `methods/DDSE/manager.py` + `configs/ddse.py`;5 头等权
- 共用:`utils/hinge_loss.py`(margin=0.15×|id_i−id_j|,基于类别 id)
- 关键适配:
  - 序列长度:DLF 加 MIntRec 分支(text 30/video 230/audio 480,用 args.text_seq_len 注入);**DDSE 的 TCSSM(MMBiMamba)要求双流等长,a/v 投影后 F.interpolate 到文本长度**(等价论文 aligned 模式),len_v/len_a 同 len_l
  - TransformerEncoder 复用 MIntRec 自带 fairseq 式实现;BertTextEncoder → MIntRec BERTEncoder
  - 输出头 1 维 → num_labels;L1 → CE;管理器硬编码 .cuda() 移除
  - 超参论文忠实(两方法 lr 1e-4,batch:DLF 16 / DDSE 32)
- 冒烟(2 epochs):DLF acc 30.6/f1 8.9(loss 18.9→14.5);DDSE acc 18.9/f1 3.0(loss 15.3→14.5)——两方法 dropout 重、头多,2 epoch 未热,链路已验证
- 运行注意:均需 `--num_workers 0`

## 待办
- [ ] 五方法全量训练(text/misa/mult/mag_bert/tcs_mamba/dlf/ddse,确认后启动)
- [ ] 结果与论文指标对比校验



