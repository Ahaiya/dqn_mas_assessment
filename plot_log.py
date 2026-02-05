"""
Training Log Visualizer
=======================
自动读取 logs/ 目录下最新的 CSV 日志并绘制监控图表。
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys

# 设置中文字体 (防止乱码，根据系统调整，Windows通常是SimHei, Mac是Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_latest_log_file(log_dir="logs"):
    """获取 logs 目录下最新的 csv 文件"""
    files = glob.glob(os.path.join(log_dir, "train_log_*.csv"))
    # 或者是 final_log_*.csv，根据您的命名习惯调整
    if not files:
        files = glob.glob(os.path.join(log_dir, "*.csv"))

    if not files:
        print(f"❌ '{log_dir}' 目录下没有找到 CSV 日志文件。")
        return None

    # 按修改时间排序
    latest_file = max(files, key=os.path.getmtime)
    print(f" 读取最新日志: {latest_file}")
    return latest_file


def plot_metrics(csv_path):
    if not csv_path: return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 创建画布 (3行1列)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f"Training Metrics: {os.path.basename(csv_path)}", fontsize=16)

    # Window Size for Moving Average
    window = 20 if len(df) > 100 else 5

    # ------------------------------------------------
    # Subplot 1: Reward (Accuracy - Efficiency)
    # ------------------------------------------------
    ax1 = axes[0]
    ax1.plot(df['episode'], df['reward'], alpha=0.3, color='gray', label='Raw Reward')
    # 计算移动平均
    df['reward_ma'] = df['reward'].rolling(window=window, min_periods=1).mean()
    ax1.plot(df['episode'], df['reward_ma'], color='blue', linewidth=2, label=f'MA({window})')

    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Trend (Higher is Better)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ------------------------------------------------
    # Subplot 2: Loss (DQN Convergence)
    # ------------------------------------------------
    ax2 = axes[1]
    # Loss 可能有空值 (Warmup 阶段)，填充为 0 或不画
    loss_data = df['loss'].fillna(0)
    ax2.plot(df['episode'], loss_data, color='red', alpha=0.6, label='Loss')

    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Loss (Lower is Better)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    # 如果 Loss 爆发，限制一下 Y 轴范围方便看细节
    if loss_data.max() > 1.0:
        ax2.set_ylim(0, min(loss_data.max(), 5.0))

    # ------------------------------------------------
    # Subplot 3: Rounds & Epsilon
    # ------------------------------------------------
    ax3 = axes[2]
    ax3.bar(df['episode'], df['rounds'], color='green', alpha=0.3, label='Rounds', width=1.0)
    ax3.set_ylabel('Debate Rounds')

    # 双 Y 轴画 Epsilon
    ax3_right = ax3.twinx()
    ax3_right.plot(df['episode'], df['epsilon'], color='orange', linestyle='--', label='Epsilon')
    ax3_right.set_ylabel('Exploration Rate (Epsilon)')

    ax3.set_title('Debate Rounds & Exploration Rate')
    # 合并图例
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_right.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc='upper right')

    ax3.set_xlabel('Episode')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图片
    img_path = csv_path.replace('.csv', '_plot.png')
    plt.savefig(img_path)
    print(f" 图表已保存至: {img_path}")
    plt.show()


if __name__ == "__main__":
    # 允许命令行传参: python plot_log.py logs/my_log.csv
    target_file = sys.argv[1] if len(sys.argv) > 1 else None

    if not target_file:
        target_file = get_latest_log_file()

    plot_metrics(target_file)