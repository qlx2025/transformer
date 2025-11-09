import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys

# -------------------------- 中文显示配置 --------------------------
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# ------------------------------------------------------------------

# 添加src目录到Python路径（确保导入正常）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from model import Transformer
from data_utils import load_iwslt_data, create_data_loaders, create_masks
from config import Config


class Trainer:
    """Transformer训练器：封装训练/验证/测试/日志/绘图功能"""

    def __init__(self, config, model, train_loader, val_loader, test_loader, tokenizer):
        self.config = config          # 配置实例
        self.model = model.to(config.device)  # 模型（迁移到GPU/CPU）
        self.train_loader = train_loader    # 训练集DataLoader
        self.val_loader = val_loader        # 验证集DataLoader
        self.test_loader = test_loader      # 测试集DataLoader
        self.tokenizer = tokenizer          # Tokenizer实例

        # 损失函数（忽略pad_token，启用标签平滑）
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # pad_token的索引（<pad>在词汇表中是第0位）
            label_smoothing=config.label_smoothing  # 标签平滑（正则化）
        )

        # 优化器（Adam）
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay  # 权重衰减（正则化）
        )

        # 学习率调度器（验证损失停止下降时降低学习率）
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',    # 基于验证损失（越小越好）
            factor=0.5,    # 学习率变为原来的0.5倍
            patience=2,    # 连续2个epoch无改善则调整
            verbose=True,  # 打印学习率调整信息
            min_lr=1e-6    # 最小学习率
        )

        # 训练记录（用于后续绘图）
        self.train_losses = []      # 训练损失
        self.val_losses = []        # 验证损失
        self.test_losses = []       # 测试损失
        self.bleu_scores = []       # BLEU分数
        self.learning_rates = []    # 学习率变化
        self.best_bleu = 0          # 最佳BLEU分数（用于保存最佳模型）
        self.best_val_loss = float('inf')  # 最佳验证损失

    def train_epoch(self):
        """训练一个epoch（单次遍历训练集）"""
        self.model.train()  # 切换到训练模式（启用dropout/batchnorm）
        epoch_loss = 0      # 记录整个epoch的损失

        # 进度条（显示训练进度）
        progress_bar = tqdm(self.train_loader, desc=f"训练Epoch [{len(self.train_losses)+1}/{self.config.epochs}]")
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            # 数据迁移到GPU/CPU
            src = src.to(self.config.device, non_blocking=True)
            tgt = tgt.to(self.config.device, non_blocking=True)

            # 创建掩码（src_mask过滤pad，tgt_mask过滤pad+未来token）
            src_mask, tgt_mask = create_masks(src, tgt)

            # 前向传播：输入tgt[:, :-1]（去掉最后一个token），预测tgt[:, 1:]（去掉第一个token）
            output = self.model(
                src=src,
                tgt=tgt[:, :-1],  # 目标输入（不含eos）
                src_mask=src_mask,
                tgt_mask=tgt_mask[:, :, :-1, :-1]  # 调整掩码维度（匹配tgt[:, :-1]）
            )

            # 计算损失：将output和tgt_output展平为2D（batch_size*seq_len, vocab_size）
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)  # (batch_size*(seq_len-1), vocab_size)
            tgt_output = tgt[:, 1:].contiguous().view(-1)     # (batch_size*(seq_len-1), )

            loss = self.criterion(output, tgt_output)

            # 反向传播 + 梯度裁剪（防止梯度爆炸）
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()             # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)  # 梯度裁剪
            self.optimizer.step()       # 更新参数

            # 累加损失，更新进度条
            epoch_loss += loss.item()
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # 定期清理GPU缓存（避免显存溢出）
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 计算整个epoch的平均损失
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        self.learning_rates.append(current_lr)
        return avg_loss

    def validate(self, loader, is_test=False):
        """验证/测试模型（无梯度计算）"""
        self.model.eval()  # 切换到评估模式（禁用dropout/batchnorm）
        epoch_loss = 0      # 记录整个epoch的损失

        with torch.no_grad():  # 禁用梯度计算（节省显存+加速）
            progress_bar = tqdm(loader, desc="测试" if is_test else "验证")
            for src, tgt in progress_bar:
                # 数据迁移到GPU/CPU
                src = src.to(self.config.device, non_blocking=True)
                tgt = tgt.to(self.config.device, non_blocking=True)

                # 创建掩码
                src_mask, tgt_mask = create_masks(src, tgt)

                # 前向传播
                output = self.model(
                    src=src,
                    tgt=tgt[:, :-1],
                    src_mask=src_mask,
                    tgt_mask=tgt_mask[:, :, :-1, :-1]
                )

                # 计算损失
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                tgt_output = tgt[:, 1:].contiguous().view(-1)
                loss = self.criterion(output, tgt_output)

                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 计算平均损失
        avg_loss = epoch_loss / len(loader)
        if is_test:
            self.test_losses.append(avg_loss)
            return avg_loss
        else:
            self.val_losses.append(avg_loss)
            # 计算BLEU分数（评估翻译质量）
            bleu_score = self.calculate_bleu()
            self.bleu_scores.append(bleu_score)
            return avg_loss, bleu_score

    def calculate_bleu(self):
        """计算BLEU分数（简化版，适配小批量验证集）"""
        self.model.eval()
        all_predictions = []  # 模型预测结果
        all_references = []   # 真实标签（参考翻译）

        with torch.no_grad():
            for src, tgt in self.val_loader:
                # 限制批量大小（避免显存溢出）
                if len(src) > 16:
                    src = src[:16]
                    tgt = tgt[:16]

                # 数据迁移到GPU/CPU
                src = src.to(self.config.device, non_blocking=True)
                tgt = tgt.to(self.config.device, non_blocking=True)
                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # 源序列掩码

                # 批量翻译（生成预测结果）
                predictions = self.translate_batch(src, src_mask)

                # 收集参考翻译（真实标签）
                references = []
                for i in range(len(tgt)):
                    ref_text = self.tokenizer.decode_tgt(tgt[i])  # 解码真实标签
                    references.append([ref_text])  # BLEU要求参考翻译是列表的列表

                # 累加结果
                all_predictions.extend(predictions)
                all_references.extend(references)

                # 仅计算前2个批次（节省时间）
                if len(all_predictions) >= 32:
                    break

        # 计算简化版BLEU分数（1-gram匹配率，适配小规模数据）
        return self.simple_bleu(all_predictions, all_references)

    def simple_bleu(self, predictions, references):
        """简化版BLEU分数计算（基于1-gram精确度）"""
        total_score = 0.0
        valid_count = 0

        for pred, refs in zip(predictions, references):
            # 分词（预测结果和参考翻译）
            pred_words = pred.split()
            ref_words = refs[0].split()

            # 跳过空句子
            if len(pred_words) == 0 or len(ref_words) == 0:
                continue

            # 计算1-gram匹配数（预测词在参考词中的数量）
            matches = len(set(pred_words) & set(ref_words))
            precision = matches / len(pred_words)  # 精确度
            total_score += precision * 100  # 转为百分比
            valid_count += 1

        # 返回平均分数（无平滑，适用于训练监控）
        return total_score / valid_count if valid_count > 0 else 0.0

    def translate_batch(self, src, src_mask, max_len=60):
        """批量翻译（推理时用：从src生成目标语言序列）"""
        self.model.eval()
        batch_size = src.shape[0]

        # 1. 编码源序列（得到记忆向量）
        memory = self.model.encode(src, src_mask)

        # 2. 初始化目标序列：以<sos>（索引1）开头
        tgt_indices = torch.ones(batch_size, 1).fill_(1).long().to(self.config.device)

        # 3. 逐词生成目标序列（直到生成<eos>或达到max_len）
        for _ in range(max_len - 1):  # 已包含<sos>，剩余max_len-1个token
            # 创建目标序列掩码（掩码自注意力用）
            tgt_mask = create_masks(tgt_indices, tgt_indices)[1]

            # 解码当前目标序列
            output = self.model.decode(
                tgt=tgt_indices,
                memory=memory,
                tgt_mask=tgt_mask
            )

            # 预测下一个词（取最后一个位置的概率最大的词）
            next_word_logits = output[:, -1, :]  # (batch_size, vocab_size)
            next_word = next_word_logits.argmax(dim=-1)  # (batch_size, )

            # 将预测词添加到目标序列
            tgt_indices = torch.cat([tgt_indices, next_word.unsqueeze(1)], dim=1)

            # 若所有序列都生成了<eos>（索引2），提前停止
            if (next_word == 2).all():
                break

        # 4. 解码目标序列索引为文本
        translations = []
        for seq in tgt_indices:
            translations.append(self.tokenizer.decode_tgt(seq.cpu()))  # 迁移到CPU解码

        return translations

    def show_translation_examples(self, epoch):
        """显示翻译示例（每5个epoch显示一次，直观查看翻译质量）"""
        self.model.eval()
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}] 翻译示例（英→德）")
        print(f"{'='*60}")

        with torch.no_grad():
            for src, tgt in self.val_loader:
                if len(src) > 0:
                    # 数据迁移到GPU/CPU
                    src = src.to(self.config.device, non_blocking=True)
                    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

                    # 生成翻译
                    translations = self.translate_batch(src[:3], src_mask[:3])  # 取前3个样本

                    # 打印示例
                    for i in range(min(3, len(src))):
                        src_text = self.tokenizer.decode_src(src[i].cpu())  # 源文本（英文）
                        true_text = self.tokenizer.decode_tgt(tgt[i].cpu())  # 真实翻译（德文）
                        pred_text = translations[i]  # 模型预测（德文）

                        print(f"源文（英）: {src_text}")
                        print(f"真实（德）: {true_text}")
                        print(f"预测（德）: {pred_text}")
                        print(f"{'-'*60}")
                    break  # 仅显示一个批次

    def plot_training_curves(self):
        """绘制训练曲线（损失+BLEU+学习率）"""
        plt.figure(figsize=(18, 5))

        # 1. 损失曲线（训练+验证+测试）
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='训练损失', linewidth=2, color='#1f77b4')
        plt.plot(self.val_losses, label='验证损失', linewidth=2, color='#ff7f0e')
        if self.test_losses:
            plt.plot(self.test_losses, label='测试损失', linewidth=2, color='#2ca02c')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('训练/验证/测试损失曲线', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 2. BLEU分数曲线
        plt.subplot(1, 3, 2)
        plt.plot(self.bleu_scores, label='BLEU分数', linewidth=2, color='#d62728')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('BLEU Score (%)', fontsize=12)
        plt.title('BLEU分数变化', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 3. 学习率曲线
        plt.subplot(1, 3, 3)
        plt.plot(self.learning_rates, label='学习率', linewidth=2, color='#9467bd')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')  # 对数坐标（更易查看学习率变化）
        plt.title('学习率调度曲线', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 保存图片（高分辨率）
        plt.tight_layout()
        save_path = os.path.join(self.config.results_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 训练曲线已保存至：{save_path}")

    def plot_comprehensive_results(self):
        """绘制综合结果图（包含统计信息）"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 损失对比
        ax1.plot(self.train_losses, label='训练损失', linewidth=2, color='#1f77b4')
        ax1.plot(self.val_losses, label='验证损失', linewidth=2, color='#ff7f0e')
        if self.test_losses:
            ax1.plot(self.test_losses, label='测试损失', linewidth=2, color='#2ca02c')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('损失曲线对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. BLEU分数进度
        ax2.plot(self.bleu_scores, color='#d62728', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('BLEU Score (%)')
        ax2.set_title('BLEU分数变化')
        ax2.grid(True, alpha=0.3)

        # 3. 学习率变化
        ax3.plot(self.learning_rates, color='#9467bd', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.set_title('学习率调度')
        ax3.grid(True, alpha=0.3)

        # 4. 训练统计信息
        final_stats = {
            '最佳BLEU分数': f"{max(self.bleu_scores):.2f}%" if self.bleu_scores else "0.00%",
            '最终训练损失': f"{self.train_losses[-1]:.4f}" if self.train_losses else "0.0000",
            '最终验证损失': f"{self.val_losses[-1]:.4f}" if self.val_losses else "0.0000",
            '最终测试损失': f"{self.test_losses[-1]:.4f}" if self.test_losses else "0.0000",
            '训练轮次': f"{len(self.train_losses)}",
            '模型参数量': f"{sum(p.numel() for p in self.model.parameters()):,}",
            '设备': self.config.device.type
        }
        stats_text = '\n'.join([f"{k}: {v}" for k, v in final_stats.items()])
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=13,
                 verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('训练统计信息', fontsize=14)

        # 保存图片
        plt.tight_layout()
        save_path = os.path.join(self.config.results_dir, 'comprehensive_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 综合结果图已保存至：{save_path}")

    def train(self):
        """完整训练流程：训练→验证→测试→保存模型→绘图"""
        print("\n" + "="*60)
        print("开始训练Transformer模型")
        print("="*60)
        print(f"模型参数量：{sum(p.numel() for p in self.model.parameters()):,}")
        print(f"训练设备：{self.config.device}（{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}）")
        print(f"训练轮次：{self.config.epochs}")
        print(f"批次大小：{self.config.batch_size}")
        print(f"初始学习率：{self.config.learning_rate:.2e}")
        print("="*60 + "\n")

        start_time = time.time()  # 记录训练开始时间

        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()

            # 1. 训练一个epoch
            train_loss = self.train_epoch()

            # 2. 验证模型
            val_loss, bleu_score = self.validate(self.val_loader)

            # 3. 学习率调度（基于验证损失）
            self.scheduler.step(val_loss)

            # 4. 保存最佳模型（基于BLEU分数和验证损失双标准）
            if bleu_score > self.best_bleu:
                self.best_bleu = bleu_score
                # 保存模型权重、优化器状态、配置等（用于后续恢复训练）
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'bleu_score': bleu_score,
                    'config': self.config.__dict__,
                    'tokenizer_src_vocab': self.tokenizer.src_vocab,
                    'tokenizer_tgt_vocab': self.tokenizer.tgt_vocab,
                    'tokenizer_src_idx2token': self.tokenizer.src_idx2token,
                    'tokenizer_tgt_idx2token': self.tokenizer.tgt_idx2token
                }, os.path.join(self.config.save_dir, 'best_bleu_model.pth'))
                print(f"✅ 保存最佳BLEU模型（分数：{bleu_score:.2f}%）至：{os.path.join(self.config.save_dir, 'best_bleu_model.pth')}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'bleu_score': bleu_score,
                    'config': self.config.__dict__
                }, os.path.join(self.config.save_dir, 'best_val_loss_model.pth'))
                print(f"✅ 保存最佳验证损失模型（损失：{val_loss:.4f}）至：{os.path.join(self.config.save_dir, 'best_val_loss_model.pth')}")

            # 5. 打印当前epoch结果
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch [{epoch+1:02d}/{self.config.epochs}] 总结：")
            print(f"  训练时间：{epoch_time:.2f}s")
            print(f"  训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f}")
            print(f"  BLEU分数：{bleu_score:.2f}% | 当前学习率：{self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  最佳BLEU分数：{self.best_bleu:.2f}% | 最佳验证损失：{self.best_val_loss:.4f}")
            print("-" * 60)

            # 6. 每5个epoch显示翻译示例
            if (epoch + 1) % 5 == 0:
                self.show_translation_examples(epoch)

            # 7. 每2个epoch保存训练曲线
            if (epoch + 1) % 2 == 0:
                self.plot_training_curves()

        # 8. 训练结束后，在测试集上评估最终性能
        print("\n" + "="*60)
        print("训练完成！开始测试集评估...")
        print("="*60)
        test_loss = self.validate(self.test_loader, is_test=True)
        final_bleu = self.calculate_bleu()  # 测试集BLEU分数
        print(f"\n测试集评估结果：")
        print(f"  测试损失：{test_loss:.4f}")
        print(f"  测试集BLEU分数：{final_bleu:.2f}%")
        print("="*60)

        # 9. 最终统计与绘图
        total_time = time.time() - start_time
        print(f"\n" + "="*60)
        print("训练总总结")
        print("="*60)
        print(f"总训练时间：{total_time / 3600:.2f}小时（{total_time:.2f}秒）")
        print(f"总训练轮次：{self.config.epochs}")
        print(f"最佳验证损失：{self.best_val_loss:.4f}")
        print(f"最佳BLEU分数：{self.best_bleu:.2f}%")
        print(f"测试集损失：{test_loss:.4f}")
        print(f"测试集BLEU分数：{final_bleu:.2f}%")
        print(f"模型参数量：{sum(p.numel() for p in self.model.parameters()):,}")
        print(f"模型保存路径：{self.config.save_dir}")
        print(f"结果图表路径：{self.config.results_dir}")
        print("="*60)

        # 绘制最终综合结果图
        self.plot_comprehensive_results()

        # 保存训练日志（文本格式，方便后续查看）
        log_content = f"""
Transformer训练日志
==================
训练配置：
- 数据集：{self.config.data_name}
- 模型参数量：{sum(p.numel() for p in self.model.parameters()):,}
- 训练轮次：{self.config.epochs}
- 批次大小：{self.config.batch_size}
- 初始学习率：{self.config.learning_rate:.2e}
- 训练设备：{self.config.device}（{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}）
- 训练时间：{total_time / 3600:.2f}小时（{total_time:.2f}秒）

训练结果：
- 最终训练损失：{self.train_losses[-1]:.4f}
- 最佳验证损失：{self.best_val_loss:.4f}
- 最终验证损失：{self.val_losses[-1]:.4f}
- 测试集损失：{test_loss:.4f}
- 最佳BLEU分数：{self.best_bleu:.2f}%
- 测试集BLEU分数：{final_bleu:.2f}%

模型保存：
- 最佳BLEU模型：{os.path.join(self.config.save_dir, 'best_bleu_model.pth')}
- 最佳验证损失模型：{os.path.join(self.config.save_dir, 'best_val_loss_model.pth')}
"""
        log_path = os.path.join(self.config.results_dir, 'training_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        print(f"✅ 训练日志已保存至：{log_path}")


def main():
    """主函数：初始化配置→加载数据→创建模型→开始训练"""
    # 1. 初始化配置（自动检测GPU）
    config = Config()

    # 2. 加载数据集（解压→拆分→编码）
    print("\n开始加载IWSLT2017数据集...")
    train_dataset, val_dataset, test_dataset, tokenizer = load_iwslt_data(config)

    # 3. 创建数据加载器（适配GPU多线程）
    print("\n创建数据加载器...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, config.batch_size
    )

    # 4. 初始化Transformer模型
    print("\n初始化Transformer模型...")
    model = Transformer(
        src_vocab_size=tokenizer.src_vocab_size,
        tgt_vocab_size=tokenizer.tgt_vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        d_ff=config.dim_feedforward,
        max_len=config.max_length,
        dropout=config.dropout,
        activation=config.activation
    )

    # 打印模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量：{total_params:,}（可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}）")

    # 5. 初始化训练器并开始训练
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        tokenizer=tokenizer
    )
    trainer.train()


if __name__ == "__main__":
    # 解决matplotlib在Linux无GUI的问题
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端（仅保存图片，不显示）
    main()