#!/usr/bin/env python3
"""
主运行脚本 - 执行完整实验流程
"""

import os
import sys
import argparse

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser(description='Transformer英德翻译实验')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'ablation', 'all'],
                        help='运行模式: train(训练), ablation(消融实验), all(全部)')
    parser.add_argument('--check-env', action='store_true',
                        help='检查环境配置')

    args = parser.parse_args()

    print("=" * 60)
    print("Transformer英德翻译实验")
    print("=" * 60)

    # 检查环境
    if args.check_env:
        from environment_check import check_environment
        check_environment()
        return

    # 创建必要目录
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./data", exist_ok=True)

    # 根据模式执行
    if args.mode in ['train', 'all']:
        print("\n🚀 开始训练主模型...")
        try:
            from src.train import main as train_main
            train_main()
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()

    if args.mode in ['ablation', 'all']:
        print("\n🔬 开始消融实验...")
        try:
            from src.ablation import main as ablation_main
            ablation_main()
        except Exception as e:
            print(f"❌ 消融实验失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 实验完成!")
    print("结果保存在 ./results/ 目录")


if __name__ == "__main__":
    main()