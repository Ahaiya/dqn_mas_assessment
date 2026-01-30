"""
Phase 5: Offline RL Training (ASAP Dataset) - With Monitoring
=============================================================
功能增强：
1. 记录每轮的 Reward 和 Loss。
2. 实时保存训练日志到 CSV。
3. 训练结束后自动绘制 Loss/Reward 曲线图。
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from langchain_core.runnables import RunnableConfig

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workflow.graph import mas_graph
from workflow.dqn_node import global_dqn_agent
from core.loaders.asap_loader import ASAPLoader

# ==========================================
# 路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_submissions", "training_set_rel3.tsv")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata", "asap_context.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")  # 新增日志目录

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)


def calculate_reward(final_state, ground_truth_score):
    """奖励函数"""
    reviews = final_state.get("reviews", [])
    if not reviews:
        return -1.0, 0.0

    last_reviews = reviews[-3:]
    if not last_reviews:
        return 0.0, 0.0

    avg_score = np.mean([r.overall_score for r in last_reviews])
    diff = abs(avg_score - ground_truth_score)

    accuracy_reward = 1.0 - diff
    rounds = final_state.get("current_round", 1)
    efficiency_penalty = 0.05 * (rounds - 1)

    total_reward = accuracy_reward - efficiency_penalty

    print(f"   🎯 Truth: {ground_truth_score:.2f} | Agents: {avg_score:.2f} | Diff: {diff:.2f}")
    print(f"   💰 Reward: {total_reward:.4f}")

    return total_reward, avg_score


def plot_metrics(metrics_df, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    # 1. Reward 曲线
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df['episode'], metrics_df['reward'], label='Reward', color='blue', alpha=0.6)
    # 绘制移动平均线 (Window=5)
    if len(metrics_df) >= 5:
        plt.plot(metrics_df['episode'], metrics_df['reward'].rolling(5).mean(), label='Avg Reward (5)', color='red',
                 linewidth=2)
    plt.title("Training Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    # 2. Loss 曲线
    plt.subplot(1, 2, 2)
    # 过滤掉 None 的 Loss
    loss_data = metrics_df[metrics_df['loss'].notna()]
    if not loss_data.empty:
        plt.plot(loss_data['episode'], loss_data['loss'], label='Loss', color='orange')
        plt.title("DQN Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'No Loss Data Yet', ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 监控图表已保存: {save_path}")
    plt.close()


def train(episodes=50):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f">>> 🚀 启动 ASAP 训练 (Episodes: {episodes}) | Session: {timestamp}")

    try:
        loader = ASAPLoader(tsv_path=DATA_PATH, metadata_path=METADATA_PATH)
        loader.load_dataset()
    except FileNotFoundError as e:
        print(f"❌ 初始化失败: {e}")
        return

    train_indices = loader.get_split_indices('train')
    global_dqn_agent.policy_net.train()

    # 📊 监控数据容器
    metrics_log = []

    for i in range(episodes):
        print(f"\n🎬 Episode {i + 1}/{episodes}")
        rand_idx = np.random.choice(train_indices)
        subject, gt_score = loader.get_subject_by_index(rand_idx)

        print(f"   📝 ID: {subject.subject_id} (Set {subject.metadata['set_id']})")

        state = {"submission": subject, "reviews": [], "current_round": 1}

        episode_reward = 0.0
        episode_loss = None
        agent_score = 0.0

        try:
            final_state = mas_graph.invoke(
                state,
                config=RunnableConfig(run_name=f"Train_Ep_{i}")
            )

            # 计算奖励
            episode_reward, agent_score = calculate_reward(final_state, gt_score)

            # 梯度更新
            loss = global_dqn_agent.update_policy(batch_size=16)
            if loss is not None:
                episode_loss = loss
                print(f"   🔥 Loss: {loss:.4f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

        # 📝 记录本轮数据
        metrics_log.append({
            "episode": i + 1,
            "subject_id": subject.subject_id,
            "set_id": subject.metadata['set_id'],
            "ground_truth": gt_score,
            "agent_score": agent_score,
            "reward": episode_reward,
            "loss": episode_loss,
            "rounds": state.get("current_round", 1)
        })

        # 每 10 轮保存一次 CSV，防止中断丢失
        if (i + 1) % 10 == 0:
            df = pd.DataFrame(metrics_log)
            csv_path = os.path.join(LOG_DIR, f"training_log_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            print(f"💾 进度已保存: {csv_path}")

    # 🏁 训练结束
    # 1. 保存模型
    model_path = os.path.join(BASE_DIR, "core", "dqn_weights_asap.pth")
    global_dqn_agent.save(model_path)
    print(f"\n💾 模型权重已保存: {model_path}")

    # 2. 保存最终日志
    df = pd.DataFrame(metrics_log)
    csv_path = os.path.join(LOG_DIR, f"training_log_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    # 3. 绘制曲线
    plot_path = os.path.join(LOG_DIR, f"training_curve_{timestamp}.png")
    plot_metrics(df, plot_path)


if __name__ == "__main__":
    # 建议至少跑 50 轮以观察曲线变化
    train(episodes=50)