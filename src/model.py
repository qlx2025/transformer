import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码层：为输入添加位置信息（Transformer无递归结构，需显式编码位置）"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码（max_len, d_model）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # 位置编码公式：PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(...)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)，适配batch维度

        # 注册为缓冲区（不参与梯度更新）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        输入：x -> (seq_len, batch_size, d_model) 或 (batch_size, seq_len, d_model)
        输出：添加位置编码后的x（同输入维度）
        """
        # 适配两种维度顺序（batch_first=True/False）
        if x.dim() == 3 and x.shape[0] != 1 and x.shape[1] != 1:
            # 若输入为 (batch_size, seq_len, d_model)，调整位置编码维度
            pe = self.pe[:x.shape[1], :].transpose(0, 1)  # (1, seq_len, d_model) → (seq_len, 1, d_model)
            x = x + pe.expand(x.shape[0], -1, -1)  # (batch_size, seq_len, d_model)
        else:
            x = x + self.pe[:x.size(0), :]  # (seq_len, batch_size, d_model)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制：将注意力分为多个头，并行计算后拼接"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model必须是nhead的整数倍"

        self.d_model = d_model        # 模型总维度
        self.nhead = nhead            # 注意力头数
        self.d_k = d_model // nhead   # 每个头的维度

        # 线性变换层（Q, K, V 共享权重，输出维度均为d_model）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 输出线性变换层
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)  # 缩放因子（避免分数过大导致softmax饱和）

    def forward(self, query, key, value, mask=None):
        """
        输入：
            query: (batch_size, seq_len_q, d_model) → 查询
            key: (batch_size, seq_len_k, d_model) → 键
            value: (batch_size, seq_len_v, d_model) → 值（seq_len_k = seq_len_v）
            mask: (batch_size, 1, seq_len_q, seq_len_k) → 掩码（可选）
        输出：
            output: (batch_size, seq_len_q, d_model) → 注意力输出
            attn_weights: (batch_size, nhead, seq_len_q, seq_len_k) → 注意力权重（用于可视化）
        """
        batch_size = query.size(0)

        # 1. 线性变换 + 分头（batch_size, seq_len, d_model）→ (batch_size, nhead, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 2. 计算注意力分数：Q@K^T / sqrt(d_k) → (batch_size, nhead, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 3. 应用掩码（mask为0的位置分数设为-1e9，softmax后接近0）
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # 4. 计算注意力权重（softmax）+ dropout
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, nhead, seq_len_q, seq_len_k)
        attn_weights = self.dropout(attn_weights)

        # 5. 注意力加权求和：attn_weights@V → (batch_size, nhead, seq_len_q, d_k)
        context = torch.matmul(attn_weights, V)

        # 6. 拼接所有头的输出 → (batch_size, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 7. 输出线性变换
        output = self.w_o(context)
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络：每个位置独立的两层全连接网络（FFN(x) = max(0, xW1 + b1)W2 + b2）"""

    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # 第一层：d_model → d_ff
        self.linear2 = nn.Linear(d_ff, d_model)  # 第二层：d_ff → d_model
        self.dropout = nn.Dropout(dropout)
        # 激活函数（支持relu/gelu）
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        """输入/输出：(batch_size, seq_len, d_model)"""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer编码器层：自注意力 + 位置前馈网络 + 残差连接 + 层归一化"""

    def __init__(self, d_model, nhead, d_ff, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)  # 自注意力
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)  # 前馈网络

        # 层归一化（Pre-LN结构：归一化在子层之前）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        输入：
            src: (batch_size, seq_len, d_model) → 源序列嵌入
            src_mask: (batch_size, 1, 1, seq_len) → 源序列掩码
        输出：
            src: (batch_size, seq_len, d_model) → 编码器层输出
        """
        # 1. 自注意力子层 + 残差连接
        attn_output, _ = self.self_attn(self.norm1(src), self.norm1(src), self.norm1(src), src_mask)
        src = src + self.dropout(attn_output)

        # 2. 前馈网络子层 + 残差连接
        ff_output = self.feed_forward(self.norm2(src))
        src = src + self.dropout(ff_output)

        return src


class DecoderLayer(nn.Module):
    """Transformer解码器层：掩码自注意力 + 交叉注意力 + 位置前馈网络 + 残差连接 + 层归一化"""

    def __init__(self, d_model, nhead, d_ff, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)   # 掩码自注意力（目标序列内部）
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)  # 交叉注意力（与编码器输出交互）
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout, activation)  # 前馈网络

        # 层归一化（Pre-LN结构）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        输入：
            tgt: (batch_size, tgt_seq_len, d_model) → 目标序列嵌入
            memory: (batch_size, src_seq_len, d_model) → 编码器输出（记忆向量）
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) → 目标序列掩码（掩码自注意力用）
            memory_mask: (batch_size, 1, 1, src_seq_len) → 源序列掩码（交叉注意力用）
        输出：
            tgt: (batch_size, tgt_seq_len, d_model) → 解码器层输出
        """
        # 1. 掩码自注意力子层 + 残差连接
        attn_output, _ = self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt), tgt_mask)
        tgt = tgt + self.dropout(attn_output)

        # 2. 交叉注意力子层 + 残差连接（query来自解码器，key/value来自编码器）
        cross_output, _ = self.cross_attn(self.norm2(tgt), self.norm2(memory), self.norm2(memory), memory_mask)
        tgt = tgt + self.dropout(cross_output)

        # 3. 前馈网络子层 + 残差连接
        ff_output = self.feed_forward(self.norm3(tgt))
        tgt = tgt + self.dropout(ff_output)

        return tgt


class TransformerEncoder(nn.Module):
    """Transformer编码器：词嵌入 + 位置编码 + 多层编码器层 + 层归一化"""

    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff,
                 max_len=5000, dropout=0.1, activation="relu"):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model  # 嵌入层/隐藏层维度

        # 词嵌入层（vocab_size → d_model）
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        # 多层编码器层（堆叠num_layers个EncoderLayer）
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        # 最终层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        """
        输入：
            src: (batch_size, seq_len) → 源序列索引（未嵌入）
            src_mask: (batch_size, 1, 1, seq_len) → 源序列掩码
        输出：
            out: (batch_size, seq_len, d_model) → 编码器最终输出
        """
        # 1. 词嵌入 + 缩放（避免嵌入向量过大）
        x = self.embedding(src) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        # 2. 添加位置编码
        x = self.pos_encoding(x)
        # 3. 经过多层编码器层
        for layer in self.layers:
            x = layer(x, src_mask)
        # 4. 最终层归一化
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """Transformer解码器：词嵌入 + 位置编码 + 多层解码器层 + 层归一化 + 输出投影"""

    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff,
                 max_len=5000, dropout=0.1, activation="relu"):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model  # 嵌入层/隐藏层维度

        # 词嵌入层（vocab_size → d_model）
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码层
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        # 多层解码器层（堆叠num_layers个DecoderLayer）
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        # 最终层归一化
        self.norm = nn.LayerNorm(d_model)
        # 输出投影层（d_model → vocab_size，预测每个位置的词）
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        输入：
            tgt: (batch_size, tgt_seq_len) → 目标序列索引（未嵌入）
            memory: (batch_size, src_seq_len, d_model) → 编码器输出
            tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) → 目标序列掩码
            memory_mask: (batch_size, 1, 1, src_seq_len) → 源序列掩码
        输出：
            out: (batch_size, tgt_seq_len, vocab_size) → 解码器最终输出（未softmax）
        """
        # 1. 词嵌入 + 缩放
        x = self.embedding(tgt) * math.sqrt(self.d_model)  # (batch_size, tgt_seq_len, d_model)
        # 2. 添加位置编码
        x = self.pos_encoding(x)
        # 3. 经过多层解码器层
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        # 4. 最终层归一化
        x = self.norm(x)
        # 5. 输出投影（预测词索引）
        output = self.output_projection(x)
        return output


class Transformer(nn.Module):
    """完整Transformer模型：编码器 + 解码器（端到端翻译）"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1, activation="relu"):
        super(Transformer, self).__init__()

        # 编码器和解码器实例化
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            activation=activation
        )

        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            activation=activation
        )

        # 初始化模型参数（Xavier均匀分布）
        self._reset_parameters()

        # 打印模型初始化信息
        print(f"Transformer模型初始化完成:")
        print(f"  - 源语言词汇表大小: {src_vocab_size}")
        print(f"  - 目标语言词汇表大小: {tgt_vocab_size}")
        print(f"  - 隐藏层维度: {d_model}")
        print(f"  - 注意力头数: {nhead}")
        print(f"  - 编码器层数: {num_encoder_layers}")
        print(f"  - 解码器层数: {num_decoder_layers}")
        print(f"  - 前馈网络维度: {d_ff}")
        print(f"  - Dropout比例: {dropout}")
        print(f"  - 激活函数: {activation}")

    def _reset_parameters(self):
        """参数初始化：所有可训练参数使用Xavier均匀分布"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        前向传播（训练时用）：
            src: (batch_size, src_seq_len) → 源序列索引
            tgt: (batch_size, tgt_seq_len) → 目标序列索引（输入tgt[:, :-1]，预测tgt[:, 1:]）
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            memory_mask: 编码器输出掩码
        输出：
            out: (batch_size, tgt_seq_len-1, vocab_size) → 预测输出（未softmax）
        """
        # 编码器编码源序列 → (batch_size, src_seq_len, d_model)
        memory = self.encoder(src, src_mask)
        # 解码器解码目标序列 → (batch_size, tgt_seq_len, vocab_size)
        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return output

    def encode(self, src, src_mask=None):
        """单独编码源序列（推理时用）"""
        return self.encoder(src, src_mask)

    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """单独解码目标序列（推理时用）"""
        return self.decoder(tgt, memory, tgt_mask, memory_mask)