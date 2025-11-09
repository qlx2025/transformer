import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import ssl
import re
import tarfile
import random

# -------------------------- 基础配置 --------------------------
# 解决SSL证书问题（下载相关依赖时用）
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

DATASET_LOCAL_DIR = "./iwslt2017_dataset"  # 数据集解压目录（Linux相对路径）
SPLIT_RATIO = (0.8, 0.1, 0.1)  # 训练集:验证集:测试集 = 8:1:1
DATASET_TRAIN_EN = None  # 全局变量：train.en路径（解压后自动赋值）
DATASET_TRAIN_DE = None  # 全局变量：train.de路径（解压后自动赋值）


class BilingualTokenizer:
    """双语Tokenizer（本地实现，无预训练依赖，适配英/德）"""

    def __init__(self, src_lang="en", tgt_lang="de", min_freq=2):
        self.src_lang = src_lang  # 源语言（英文）
        self.tgt_lang = tgt_lang  # 目标语言（德文）
        self.min_freq = min_freq  # 最小词频（过滤低频词）

        # 特殊Token（pad:填充，sos:起始，eos:结束，unk:未登录词）
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'

        # 初始化词汇表（词→索引）和反向词汇表（索引→词）
        self.src_vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.tgt_vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.src_idx2token = {idx: token for idx, token in enumerate(self.special_tokens)}
        self.tgt_idx2token = {idx: token for idx, token in enumerate(self.special_tokens)}

        # 词汇表大小（初始为特殊Token数量）
        self.src_vocab_size = len(self.special_tokens)
        self.tgt_vocab_size = len(self.special_tokens)

        # 分词函数（适配英/德）
        self.src_tokenizer = self._simple_tokenize_en
        self.tgt_tokenizer = self._simple_tokenize_de

    def _simple_tokenize_en(self, text):
        """英文分词：小写→去标点→空格分割"""
        text = text.strip().lower()
        text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
        return text.split()  # 空格分割

    def _simple_tokenize_de(self, text):
        """德文分词：小写→保留äöüß→去标点→空格分割"""
        text = text.strip().lower()
        text = re.sub(r'[^\w\säöüß]', '', text)  # 保留德文特殊字符
        return text.split()  # 空格分割

    def build_vocab_from_data(self, dataset, max_vocab_size=10000):
        """从数据集构建词汇表（仅用训练集，避免数据泄露）"""
        src_word_freq = {}  # 源语言词频统计
        tgt_word_freq = {}  # 目标语言词频统计

        print(f"构建词汇表（min_freq={self.min_freq}，max_vocab_size={max_vocab_size}）...")
        for item in tqdm(dataset, desc="统计词频"):
            # 源语言文本处理
            src_text = item['translation'][self.src_lang].strip()
            src_tokens = self.src_tokenizer(src_text)
            for token in src_tokens:
                src_word_freq[token] = src_word_freq.get(token, 0) + 1

            # 目标语言文本处理
            tgt_text = item['translation'][self.tgt_lang].strip()
            tgt_tokens = self.tgt_tokenizer(tgt_text)
            for token in tgt_tokens:
                tgt_word_freq[token] = tgt_word_freq.get(token, 0) + 1

        # 过滤低频词（仅保留词频≥min_freq的词）
        src_filtered = {w: f for w, f in src_word_freq.items() if f >= self.min_freq}
        tgt_filtered = {w: f for w, f in tgt_word_freq.items() if f >= self.min_freq}

        # 选择高频词（限制词汇表最大大小）
        src_common = sorted(src_filtered.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]
        tgt_common = sorted(tgt_filtered.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]

        # 更新源语言词汇表
        for word, _ in src_common:
            if word not in self.src_vocab:
                self.src_vocab[word] = self.src_vocab_size
                self.src_idx2token[self.src_vocab_size] = word
                self.src_vocab_size += 1

        # 更新目标语言词汇表
        for word, _ in tgt_common:
            if word not in self.tgt_vocab:
                self.tgt_vocab[word] = self.tgt_vocab_size
                self.tgt_idx2token[self.tgt_vocab_size] = word
                self.tgt_vocab_size += 1

        # 打印词汇表信息
        print(f"源语言（英文）词汇表大小: {self.src_vocab_size}")
        print(f"目标语言（德文）词汇表大小: {self.tgt_vocab_size}")
        return self.src_vocab_size, self.tgt_vocab_size

    def encode_src(self, text, max_length):
        """编码源语言文本（转为Tensor，含padding/sos/eos）"""
        tokens = self.src_tokenizer(text)
        tokens = [self.sos_token] + tokens + [self.eos_token]  # 添加起始/结束符
        # 截断过长句子
        if len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.eos_token]
        # 转为索引（未登录词用unk）
        indices = [self.src_vocab.get(t, self.src_vocab[self.unk_token]) for t in tokens]
        # 填充到max_length
        while len(indices) < max_length:
            indices.append(self.src_vocab[self.pad_token])
        return torch.tensor(indices, dtype=torch.long)

    def encode_tgt(self, text, max_length):
        """编码目标语言文本（逻辑同encode_src）"""
        tokens = self.tgt_tokenizer(text)
        tokens = [self.sos_token] + tokens + [self.eos_token]
        if len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.eos_token]
        indices = [self.tgt_vocab.get(t, self.tgt_vocab[self.unk_token]) for t in tokens]
        while len(indices) < max_length:
            indices.append(self.tgt_vocab[self.pad_token])
        return torch.tensor(indices, dtype=torch.long)

    def decode_src(self, indices):
        """解码源语言Tensor（转为文本，忽略pad/sos/eos）"""
        tokens = []
        for idx in indices:
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            if idx == self.src_vocab[self.pad_token]:
                continue  # 跳过填充
            if idx == self.src_vocab[self.eos_token]:
                break     # 遇到结束符停止
            if idx == self.src_vocab[self.sos_token]:
                continue  # 跳过起始符
            tokens.append(self.src_idx2token.get(idx, self.unk_token))
        return ' '.join(tokens)

    def decode_tgt(self, indices):
        """解码目标语言Tensor（逻辑同decode_src）"""
        tokens = []
        for idx in indices:
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            if idx == self.tgt_vocab[self.pad_token]:
                continue
            if idx == self.tgt_vocab[self.eos_token]:
                break
            if idx == self.tgt_vocab[self.sos_token]:
                continue
            tokens.append(self.tgt_idx2token.get(idx, self.unk_token))
        return ' '.join(tokens)


class TranslationDataset(Dataset):
    """翻译数据集（适配PyTorch DataLoader）"""

    def __init__(self, dataset, tokenizer, max_length=60):
        self.dataset = dataset    # 原始数据集（句对列表）
        self.tokenizer = tokenizer# Tokenizer实例
        self.max_length = max_length  # 句子最大长度

    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """根据索引获取单个样本（编码为Tensor）"""
        item = self.dataset[idx]
        src_text = item['translation']['en']  # 源语言：英文
        tgt_text = item['translation']['de']  # 目标语言：德文
        # 编码为Tensor
        src_tokens = self.tokenizer.encode_src(src_text, self.max_length)
        tgt_tokens = self.tokenizer.encode_tgt(tgt_text, self.max_length)
        return src_tokens, tgt_tokens


def extract_local_dataset():
    """解压本地数据集压缩包（仅含train.en/train.de，适配Linux）"""
    # 压缩包路径（Linux项目根目录，确保压缩包在此处）
    LOCAL_ZIP_PATH = "./iwslt2017-en-de.tar.gz"

    # 检查压缩包是否存在
    if not os.path.exists(LOCAL_ZIP_PATH):
        print(f"❌ 未找到本地压缩包：{LOCAL_ZIP_PATH}")
        print("请将压缩包放在项目根目录，并重命名为 'iwslt2017-en-de.tar.gz'")
        return False

    # 创建解压目录（不存在则创建）
    os.makedirs(DATASET_LOCAL_DIR, exist_ok=True)

    # 解压tar.gz压缩包
    try:
        print(f"开始解压本地数据集：{LOCAL_ZIP_PATH}...")
        with tarfile.open(LOCAL_ZIP_PATH, 'r:gz') as tar_ref:
            tar_ref.extractall(DATASET_LOCAL_DIR)  # 解压到指定目录
        print(f"✅ 解压完成，解压目录：{DATASET_LOCAL_DIR}")

        # 搜索train.en和train.de（兼容压缩包内有子目录的情况）
        global DATASET_TRAIN_EN, DATASET_TRAIN_DE
        for root, dirs, files in os.walk(DATASET_LOCAL_DIR):
            if "train.en" in files:
                DATASET_TRAIN_EN = os.path.join(root, "train.en")
            if "train.de" in files:
                DATASET_TRAIN_DE = os.path.join(root, "train.de")
            if DATASET_TRAIN_EN and DATASET_TRAIN_DE:
                break  # 找到文件后停止搜索

        # 验证文件是否存在
        if not DATASET_TRAIN_EN or not DATASET_TRAIN_DE:
            print("❌ 解压后未找到 train.en 或 train.de 文件")
            return False
        print(f"✅ 找到训练集文件：")
        print(f"  - {DATASET_TRAIN_EN}")
        print(f"  - {DATASET_TRAIN_DE}")
        return True
    except Exception as e:
        print(f"❌ 解压失败：{str(e)}")
        return False


def parse_train_only_dataset():
    """解析训练集，拆分出验证集和测试集（8:1:1）"""
    # 获取训练集文件路径（从extract_local_dataset赋值的全局变量）
    train_en_path = DATASET_TRAIN_EN
    train_de_path = DATASET_TRAIN_DE

    # 若全局变量未赋值，使用默认路径兜底
    if not train_en_path:
        train_en_path = os.path.join(DATASET_LOCAL_DIR, "train.en")
    if not train_de_path:
        train_de_path = os.path.join(DATASET_LOCAL_DIR, "train.de")

    # 读取英-德句对
    all_pairs = []
    try:
        print(f"开始读取训练集句对...")
        with open(train_en_path, 'r', encoding='utf-8') as f_en, \
             open(train_de_path, 'r', encoding='utf-8') as f_de:
            # 读取并过滤空行
            en_lines = [line.strip() for line in f_en if line.strip()]
            de_lines = [line.strip() for line in f_de if line.strip()]
            # 取较短的长度（避免句对不匹配）
            min_len = min(len(en_lines), len(de_lines))
            # 构建句对列表
            for en, de in zip(en_lines[:min_len], de_lines[:min_len]):
                all_pairs.append({
                    'translation': {'en': en, 'de': de}
                })
        print(f"✅ 成功读取 {len(all_pairs)} 个英-德句对")
    except Exception as e:
        print(f"❌ 读取训练集失败：{str(e)}")
        return None

    # 打乱数据（固定种子，保证可复现）
    random.seed(42)
    random.shuffle(all_pairs)

    # 按比例拆分
    total = len(all_pairs)
    train_size = int(total * SPLIT_RATIO[0])
    val_size = int(total * SPLIT_RATIO[1])
    test_size = total - train_size - val_size

    # 构建数据集字典
    dataset = {
        'train': all_pairs[:train_size],
        'validation': all_pairs[train_size:train_size + val_size],
        'test': all_pairs[train_size + val_size:]
    }

    # 打印拆分结果
    print(f"✅ 数据集拆分完成：")
    print(f"  - 训练集：{len(dataset['train'])} 句对")
    print(f"  - 验证集：{len(dataset['validation'])} 句对")
    print(f"  - 测试集：{len(dataset['test'])} 句对")
    return dataset


def load_iwslt_data(config):
    """加载数据集（解压→拆分→编码→返回Dataset和Tokenizer）"""
    # 1. 解压本地压缩包
    extract_success = extract_local_dataset()
    if not extract_success:
        print("⚠️ 数据集加载失败，自动切换到示例数据（仅用于测试）")
        return create_sample_data(config)

    # 2. 解析并拆分数据集
    print("\n开始解析并拆分数据集...")
    dataset = parse_train_only_dataset()
    if dataset is None:
        print("⚠️ 数据集解析失败，自动切换到示例数据（仅用于测试）")
        return create_sample_data(config)

    # 3. 创建Tokenizer并构建词汇表（仅用训练集，避免数据泄露）
    tokenizer = BilingualTokenizer(
        src_lang="en",
        tgt_lang="de",
        min_freq=config.min_freq
    )
    tokenizer.build_vocab_from_data(dataset['train'])

    # 4. 创建PyTorch Dataset
    train_dataset = TranslationDataset(dataset['train'], tokenizer, config.max_length)
    val_dataset = TranslationDataset(dataset['validation'], tokenizer, config.max_length)
    test_dataset = TranslationDataset(dataset['test'], tokenizer, config.max_length)

    # 打印数据集最终信息
    print("\n" + "=" * 50)
    print("IWSLT2017数据集加载完成！")
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"验证集样本数：{len(val_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"源语言词汇表大小：{tokenizer.src_vocab_size}")
    print(f"目标语言词汇表大小：{tokenizer.tgt_vocab_size}")
    print("=" * 50)

    return train_dataset, val_dataset, test_dataset, tokenizer


def create_sample_data(config):
    """创建示例数据（数据集加载失败时兜底）"""
    sample_data = [
        {"translation": {"en": "the cat sits on the mat", "de": "die katze sitzt auf der matte"}},
        {"translation": {"en": "i love machine learning", "de": "ich liebe maschinelles lernen"}},
        {"translation": {"en": "the weather is beautiful today", "de": "das wetter ist heute schön"}},
        {"translation": {"en": "this is a transformer model", "de": "das ist ein transformer modell"}},
        {"translation": {"en": "natural language processing is interesting", "de": "natürliche sprachverarbeitung ist interessant"}},
    ] * 200  # 重复200次，共1000个样本

    # 创建Tokenizer并构建词汇表
    tokenizer = BilingualTokenizer(src_lang="en", tgt_lang="de", min_freq=config.min_freq)
    tokenizer.build_vocab_from_data(sample_data)

    # 拆分示例数据
    train_size = int(len(sample_data) * 0.8)
    val_size = int(len(sample_data) * 0.1)
    test_size = len(sample_data) - train_size - val_size

    # 创建Dataset
    train_dataset = TranslationDataset(sample_data[:train_size], tokenizer, config.max_length)
    val_dataset = TranslationDataset(sample_data[train_size:train_size + val_size], tokenizer, config.max_length)
    test_dataset = TranslationDataset(sample_data[train_size + val_size:], tokenizer, config.max_length)

    # 打印示例数据信息
    print(f"⚠️ 示例数据信息：")
    print(f"  - 训练集：{len(train_dataset)} 样本")
    print(f"  - 验证集：{len(val_dataset)} 样本")
    print(f"  - 测试集：{len(test_dataset)} 样本")
    print(f"  - 词汇表大小：{tokenizer.src_vocab_size}（英）/{tokenizer.tgt_vocab_size}（德）")
    return train_dataset, val_dataset, test_dataset, tokenizer


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    """创建PyTorch DataLoader（适配Linux多线程，GPU加速）"""

    def collate_fn(batch):
        """批量处理函数：将多个样本堆叠为批次Tensor"""
        src_batch = torch.stack([item[0] for item in batch])  # (batch_size, max_length)
        tgt_batch = torch.stack([item[1] for item in batch])  # (batch_size, max_length)
        return src_batch, tgt_batch

    # Linux下启用多线程（根据CPU核心数自动适配）
    num_workers = 4 if os.cpu_count() >= 4 else os.cpu_count() or 0
    print(f"数据加载器配置：batch_size={batch_size}，num_workers={num_workers}")

    # 训练集DataLoader（打乱+多线程+pin_memory）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # GPU时启用pin_memory加速
        drop_last=True  # 丢弃最后一个不完整批次
    )

    # 验证集/测试集DataLoader（不打乱+更大批次）
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # 验证/测试时批次可翻倍（节省时间）
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def create_masks(src, tgt, pad_idx=0):
    """创建注意力掩码（src_mask：过滤pad；tgt_mask：过滤pad+下三角掩码）"""
    batch_size, src_len = src.shape
    _, tgt_len = tgt.shape

    # 源语言掩码：(batch_size, 1, 1, src_len)，pad位置为False
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # 扩展维度适配注意力计算

    # 目标语言掩码：pad掩码 + 下三角掩码（防止查看未来token）
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=src.device))  # 下三角矩阵
    tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
    tgt_mask = tgt_pad_mask & tgt_sub_mask  # 合并掩码

    return src_mask, tgt_mask