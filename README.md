# Encoder-Decoder Transformer

本项目基于PyTorch框架，从零实现了**Encoder–Decoder**结构的**Transformer**模型，针对**IWSLT2017**英德平行语料库完成机器翻译任务。
（1）基于 pytorch 实现**模块化的 Transformer **模型，代码模块化高；
（2）通过消融实验量化分析了模型超参数对模型性能的影响，为同类任务的模型配置提供了参考依据；
（3）构建完整的可复现流程，包括数据预处理、模型训练与结果可视化。

---

## 1.模型架构

### Transformer 架构
本实验中Transformer 模型基于注意力机制，由 Encoder 和 Decoder 两部分组成，每个部分都有多个相同的层组成，每层包括多头注意力机制和前馈网络，包括的框架如下：
- **缩放点积注意力 (Scaled Dot-Product Attention)**
- **多头注意力 (Multi-Head Attention)**
- **基于位置的前馈网络 (Position-Wise Feed-Forward Network)**
- **残差连接与层归一化 (Residual Connections and Layer Normalization)**
- **位置编码 (Positional Encoding)**

## 2.实验设置
### 数据集
本实验采用 **IWSLT2017(EN↔DE)** 英德翻译数据集
### 训练参数设置
| 参数 | 数值 |
| --- | --- |
| 批次大小 (BatchSize) | 32 |
| 梯度累积步数 | 2 |
| 训练轮次 (Epochs) | 20 |
| 初始学习率 | 1e-4 |
| 优化器 | Adam |
| 数值稳定性参数 (eps) | 1e-9 |
| 权重衰减 (WeightDecay) | 0.0001 |
| 梯度裁剪阈值 | 1.0 |
| 标签平滑系数 | 0.1 |
| 学习率调度器 | ReduceLROnPlateau |
| 调度因子 | 0.5 |
| 调度耐心值 | 2 |
| 最小学习率 | 1e-6 |
| 随机种子 | 42 |
### 评价指标
`损失函数(Loss)` `BLEU分数`
## 3.可复现性与代码结构
### 代码结构
```bash
Transformer1/
├── src/                  # 核心代码目录
│   ├── config.py         # 配置类
│   ├── data_utils.py     # 数据集加载、预处理、Tokenizer实现
│   ├── model.py          # Encoder–Decoder Transformer核心实现
│   ├── train.py          # 训练/验证/测试全流程脚本
│   └── ablation.py       # 消融实验脚本
├── scripts/
│   ├── run.py            # 执行完整实验脚本
├── iwslt2017-en-de.tar.gz  # IWSLT2017数据集压缩包
├── checkpoints/          # 模型权重自动保存目录
├── results/              # 训练日志、损失/BLEU曲线、结果文件
└── README.md             # 项目说明文档
└── requirements.txt      # 环境配置
```
### 依赖环境
本项目依赖以下环境：
- `torch`

- `datasets`

- `transformers`

- `tqdm`

- `numpy`

- `matplotlib`

### 安装依赖
```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm==4.66.5 matplotlib==3.7.5 numpy==1.24.4 datasets==2.14.6 transformers==4.35.2
```
或
```bash
python -m pip install -r requirements.txt
```
### 硬件要求
对于GPU，NVIDIA GeForce RTX 2080 Ti（11GB 显存）及以上，或同级别GPU（如RTX 3090、Tesla V100）；对于存储预留≥20GB空间（数据集解压+模型权重+结果文件）。

### 运行命令
执行基础实验——运行train.py
```bash
CUDA_VISIBLE_DEVICES=1 python src/train.py
```
执行消融实验——运行ablation.py
```bash
CUDA_VISIBLE_DEVICES=1 python src/ablation.py
```
执行完整实验——运行run.py
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run.py
```
