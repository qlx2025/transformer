import torch
import os


class Config:
    """基础配置参数类（自动检测GPU）"""

    # 数据配置
    data_name = "iwslt2017"
    max_length = 60  # 句子最大长度
    min_freq = 2     # 词汇表最小词频

    # 模型配置
    d_model = 256          # 嵌入层/隐藏层维度
    nhead = 8              # 注意力头数
    num_encoder_layers = 4 # 编码器层数
    num_decoder_layers = 4 # 解码器层数
    dim_feedforward = 1024 # 前馈网络隐藏层维度
    dropout = 0.1          #  dropout比例
    activation = "relu"    # 激活函数

    # 训练配置
    batch_size = 32        # 批次大小（GPU可适当调大）
    accumulate_grad_batches = 2 # 梯度累积步数
    epochs = 20            # 训练轮次
    learning_rate = 1e-4   # 初始学习率
    betas = (0.9, 0.98)    # Adam优化器参数
    eps = 1e-9             # Adam数值稳定性参数
    weight_decay = 0.0001  # 权重衰减（正则化）
    clip_grad = 1.0        # 梯度裁剪阈值
    label_smoothing = 0.1  # 标签平滑系数

    # 实验配置
    seed = 42              # 随机种子（保证可复现）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测GPU/CPU

    # 保存路径（Linux相对路径）
    save_dir = "./checkpoints"   # 模型权重保存目录
    results_dir = "./results"    # 训练结果/图表保存目录

    def __init__(self):
        print("=" * 50)
        print("Transformer配置")
        print("=" * 50)
        print(f"设备: {self.device}")
        print(f"GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")
        print(f"模型规模: {self.estimate_parameters():,} 参数")

        # 创建保存目录（不存在则自动创建）
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # 设置随机种子（保证可复现）
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # 多GPU时生效
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def estimate_parameters(self):
        """估算模型参数量（仅用于初始展示）"""
        vocab_size = 8000  # 假设词汇表大小
        # 嵌入层参数（源+目标）
        embedding_params = 2 * vocab_size * self.d_model
        # 编码器参数（自注意力+前馈网络+归一化）
        encoder_attn = self.num_encoder_layers * 4 * self.d_model * self.d_model
        encoder_ffn = self.num_encoder_layers * 2 * self.d_model * self.dim_feedforward
        encoder_norm = self.num_encoder_layers * 2 * self.d_model
        # 解码器参数（自注意力+交叉注意力+前馈网络+归一化）
        decoder_attn = self.num_decoder_layers * 6 * self.d_model * self.d_model
        decoder_ffn = self.num_decoder_layers * 2 * self.d_model * self.dim_feedforward
        decoder_norm = self.num_decoder_layers * 3 * self.d_model
        # 输出层参数
        output_proj = self.d_model * vocab_size
        # 总参数量
        total = (embedding_params + encoder_attn + encoder_ffn + encoder_norm +
                 decoder_attn + decoder_ffn + decoder_norm + output_proj)
        return total


class AblationConfig(Config):
    """消融实验配置（继承基础配置，修改特定参数）"""

    def __init__(self, model_type="baseline"):
        super().__init__()  # 继承基础配置
        self.model_type = model_type

        # 根据模型类型修改配置
        if model_type == "small":
            # 小模型：减小维度和层数
            self.d_model = 128
            self.nhead = 4
            self.num_encoder_layers = 2
            self.num_decoder_layers = 2
            self.dim_feedforward = 512
        elif model_type == "large":
            # 大模型：增大维度和层数
            self.d_model = 512
            self.nhead = 8
            self.num_encoder_layers = 6
            self.num_decoder_layers = 6
            self.dim_feedforward = 2048
        elif model_type == "no_dropout":
            # 无dropout：关闭正则化
            self.dropout = 0.0
        elif model_type == "more_heads":
            # 更多注意力头
            self.nhead = 16
        elif model_type == "deep":
            # 更深层数
            self.num_encoder_layers = 8
            self.num_decoder_layers = 8

        # 打印消融实验配置
        print(f"消融实验: {model_type}")
        print(f"调整后模型规模: {self.estimate_parameters():,} 参数")