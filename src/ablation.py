import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# -------------------------- 中文显示配置 --------------------------
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
# ------------------------------------------------------------------

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 解决matplotlib在Linux无GUI的问题
import matplotlib
matplotlib.use('Agg')

# 导入自定义模块
from model import Transformer
from data_utils import load_iwslt_data, create_data_loaders, create_masks
from config import AblationConfig


class AblationStudy:
    """消融实验类：对比不同模型配置的性能（参数量、BLEU分数）"""

    def __init__(self):
        self.results = []  # 存储所有消融实验结果
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"消融实验设备：{self.device}（{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}）")

    def run_ablation(self, model_types=None):
        """运行消融实验（默认测试6种模型配置）"""
        # 定义要测试的模型类型（可自定义增减）
        if model_types is None:
            model_types = [
                "baseline",    # 基准模型（默认配置）
                "small",       # 小模型（小维度+少层数）
                "large",       # 大模型（大维度+多层数）
                "no_dropout",  # 无dropout（关闭正则化）
                "more_heads",  # 更多注意力头
                "deep"         # 更深层数
            ]

        print("=" * 70)
        print("开始Transformer消融实验")
        print("=" * 70)
        print(f"测试模型配置：{', '.join(model_types)}")
        print(f"实验设备：{self.device}")
        print(f"训练轮次：8（消融实验快速验证）")
        print("=" * 70 + "\n")

        # 遍历每种模型配置，运行实验
        for idx, model_type in enumerate(model_types, 1):
            print(f"\n🔬 实验 {idx}/{len(model_types)}：{model_type}")
            print("-" * 50)

            try:
                # 1. 初始化当前模型的配置（继承AblationConfig，自动修改对应参数）
                config = AblationConfig(model_type=model_type)
                config.epochs = 8  # 消融实验减少训练轮次（快速对比）
                config.batch_size = 32  # 统一批次大小（公平对比）

                # 2. 加载数据集（所有模型共享同一数据集，保证对比公平）
                train_dataset, val_dataset, test_dataset, tokenizer = load_iwslt_data(config)
                train_loader, val_loader, test_loader = create_data_loaders(
                    train_dataset, val_dataset, test_dataset, config.batch_size
                )

                # 3. 初始化当前配置的模型
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
                ).to(self.device)

                # 打印当前模型信息
                total_params = sum(p.numel() for p in model.parameters())
                print(f"模型参数量：{total_params:,}")
                print(f"模型配置：d_model={config.d_model}, nhead={config.nhead}, "
                      f"layers={config.num_encoder_layers}/{config.num_decoder_layers}, "
                      f"d_ff={config.dim_feedforward}, dropout={config.dropout}")

                # 4. 快速训练并获取最佳BLEU分数
                best_bleu = self.fast_train(model, config, train_loader, val_loader, tokenizer)

                # 5. 记录实验结果
                result = {
                    'model_type': model_type,
                    'parameters': total_params,
                    'd_model': config.d_model,
                    'nhead': config.nhead,
                    'num_encoder_layers': config.num_encoder_layers,
                    'num_decoder_layers': config.num_decoder_layers,
                    'd_ff': config.dim_feedforward,
                    'dropout': config.dropout,
                    'best_bleu': best_bleu,
                    'params_million': round(total_params / 1e6, 2)  # 参数量（百万）
                }
                self.results.append(result)
                print(f"✅ 实验完成：{model_type} | BLEU分数：{best_bleu:.2f}% | 参数量：{total_params:,}")

                # 清理GPU缓存（避免多模型训练显存溢出）
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ 实验失败：{model_type} | 错误信息：{str(e)[:200]}")
                continue

        # 6. 保存实验结果并绘图
        self.save_results()
        self.plot_results()

        # 7. 输出实验总结
        self.print_summary()

        return self.results

    def fast_train(self, model, config, train_loader, val_loader, tokenizer):
        """快速训练（适配消融实验，简化训练流程，聚焦性能对比）"""
        # 损失函数（忽略pad_token）
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        # 优化器（统一使用Adam，保证对比公平）
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, verbose=False
        )

        best_bleu = 0.0  # 记录最佳BLEU分数

        for epoch in range(config.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for src, tgt in train_loader:
                src = src.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)

                # 创建掩码
                src_mask, tgt_mask = create_masks(src, tgt)

                # 前向传播
                output = model(
                    src=src,
                    tgt=tgt[:, :-1],
                    src_mask=src_mask,
                    tgt_mask=tgt_mask[:, :, :-1, :-1]
                )

                # 计算损失
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                tgt_output = tgt[:, 1:].contiguous().view(-1)
                loss = criterion(output, tgt_output)

                # 反向传播 + 梯度裁剪
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                optimizer.step()

                train_loss += loss.item()

            # 验证阶段（计算BLEU分数）
            model.eval()
            current_bleu = 0.0
            with torch.no_grad():
                for src, tgt in val_loader:
                    # 限制批量大小（节省显存）
                    if len(src) > 8:
                        src = src[:8]
                        tgt = tgt[:8]

                    src = src.to(self.device, non_blocking=True)
                    tgt = tgt.to(self.device, non_blocking=True)
                    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

                    # 计算当前批次的BLEU分数
                    batch_bleu = self.calculate_bleu_batch(model, src, tgt, src_mask, tokenizer)
                    current_bleu += batch_bleu
                    break  # 仅验证1个批次（加速消融实验）

            # 平均BLEU分数（单批次）
            current_bleu = current_bleu if current_bleu == 0 else current_bleu / 1
            # 更新最佳BLEU分数
            if current_bleu > best_bleu:
                best_bleu = current_bleu

            # 学习率调度
            avg_train_loss = train_loss / len(train_loader)
            scheduler.step(avg_train_loss)

            # 打印当前epoch进度
            print(f"Epoch [{epoch+1}/{config.epochs}] | 训练损失：{avg_train_loss:.4f} | BLEU分数：{current_bleu:.2f}%")

        return best_bleu

    def calculate_bleu_batch(self, model, src, tgt, src_mask, tokenizer):
        """批量计算BLEU分数（简化版，适配消融实验）"""
        batch_size = src.shape[0]

        # 1. 编码源序列
        memory = model.encode(src, src_mask)

        # 2. 生成目标序列
        tgt_indices = torch.ones(batch_size, 1).fill_(1).long().to(self.device)  # <sos>
        for _ in range(50):  # 最大生成长度50
            tgt_mask = create_masks(tgt_indices, tgt_indices)[1]
            output = model.decode(tgt_indices, memory, tgt_mask)
            next_word = output[:, -1].argmax(dim=-1)
            tgt_indices = torch.cat([tgt_indices, next_word.unsqueeze(1)], dim=1)
            if (next_word == 2).all():  # 所有序列生成<eos>，提前停止
                break

        # 3. 解码并计算BLEU分数（1-gram匹配率）
        total_bleu = 0.0
        valid_count = 0
        for i in range(batch_size):
            pred_text = tokenizer.decode_tgt(tgt_indices[i].cpu())
            true_text = tokenizer.decode_tgt(tgt[i].cpu())

            pred_words = pred_text.split()
            true_words = true_text.split()

            if len(pred_words) == 0 or len(true_words) == 0:
                continue

            # 1-gram匹配数
            matches = len(set(pred_words) & set(true_words))
            precision = matches / len(pred_words)
            total_bleu += precision * 100
            valid_count += 1

        return total_bleu / valid_count if valid_count > 0 else 0.0

    def save_results(self):
        """保存消融实验结果（CSV格式，方便后续分析）"""
        # 创建结果目录（不存在则创建）
        os.makedirs("./results", exist_ok=True)

        # 转换结果为DataFrame并保存
        df = pd.DataFrame(self.results)
        csv_path = os.path.join("./results", "ablation_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"\n📊 消融实验结果已保存至：{csv_path}")

        # 打印结果表格（直观查看）
        print("\n消融实验结果汇总：")
        print("=" * 100)
        print(f"{'模型类型':<12} {'参数量(百万)':<12} {'BLEU分数(%)':<12} {'d_model':<8} {'nhead':<6} {'层数':<8} {'d_ff':<8} {'dropout':<8}")
        print("-" * 100)
        for result in self.results:
            print(f"{result['model_type']:<12} {result['params_million']:<12.2f} {result['best_bleu']:<12.2f} "
                  f"{result['d_model']:<8} {result['nhead']:<6} {result['num_encoder_layers']:<8} "
                  f"{result['d_ff']:<8} {result['dropout']:<8.1f}")
        print("=" * 100)

    def plot_results(self):
        """绘制消融实验结果图（BLEU分数对比+参数量对比）"""
        if not self.results:
            print("❌ 无实验结果，跳过绘图")
            return

        # 提取绘图数据
        model_types = [r['model_type'] for r in self.results]
        bleu_scores = [r['best_bleu'] for r in self.results]
        params_million = [r['params_million'] for r in self.results]

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. BLEU分数对比（柱状图）
        colors1 = plt.cm.Set3(range(len(model_types)))
        bars1 = ax1.bar(model_types, bleu_scores, color=colors1, alpha=0.8)
        ax1.set_xlabel('模型配置', fontsize=12)
        ax1.set_ylabel('BLEU分数 (%)', fontsize=12)
        ax1.set_title('消融实验 - BLEU分数对比', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签
        for bar, score in zip(bars1, bleu_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{score:.2f}%', ha='center', va='bottom', fontweight='bold')

        # 2. 参数量对比（柱状图）
        colors2 = plt.cm.Set2(range(len(model_types)))
        bars2 = ax2.bar(model_types, params_million, color=colors2, alpha=0.8)
        ax2.set_xlabel('模型配置', fontsize=12)
        ax2.set_ylabel('参数量（百万）', fontsize=12)
        ax2.set_title('消融实验 - 模型参数量对比', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签
        for bar, param in zip(bars2, params_million):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{param:.2f}M', ha='center', va='bottom', fontweight='bold')

        # 调整布局并保存
        plt.tight_layout()
        plot_path = os.path.join("./results", "ablation_study_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 消融实验图表已保存至：{plot_path}")

    def print_summary(self):
        """打印消融实验总结（找出最优配置）"""
        if not self.results:
            return

        # 找出BLEU分数最高的配置
        best_result = max(self.results, key=lambda x: x['best_bleu'])
        # 找出参数量最小但BLEU分数前3的配置（兼顾性能和效率）
        efficient_results = sorted(self.results, key=lambda x: (x['best_bleu'], -x['parameters']), reverse=True)[:3]

        print("\n🎯 消融实验总结")
        print("=" * 70)
        print(f"最佳性能配置：{best_result['model_type']}")
        print(f"  - BLEU分数：{best_result['best_bleu']:.2f}%")
        print(f"  - 参数量：{best_result['parameters']:,}（{best_result['params_million']:.2f}M）")
        print(f"  - 关键配置：d_model={best_result['d_model']}, nhead={best_result['nhead']}, "
              f"layers={best_result['num_encoder_layers']}, d_ff={best_result['d_ff']}")

        print(f"\n高效配置TOP3（性能-效率平衡）：")
        for i, res in enumerate(efficient_results, 1):
            print(f"  {i}. {res['model_type']} | BLEU：{res['best_bleu']:.2f}% | 参数量：{res['params_million']:.2f}M")
        print("=" * 70)


def main():
    """运行消融实验"""
    # 初始化消融实验
    ablation = AblationStudy()
    # 运行实验
    results = ablation.run_ablation()
    print("\n🎉 消融实验全部完成！结果已保存至 ./results 目录")


if __name__ == "__main__":
    main()