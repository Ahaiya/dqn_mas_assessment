"""
Phase 5: Offline RL Training (ASAP Dataset) - Refactored
========================================================
适配新的目录结构：分离数据与代码。
"""
import sys
import os
import numpy as np
from langchain_core.runnables import RunnableConfig

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from workflow.graph import mas_graph
from workflow.dqn_node import global_dqn_agent
# 🌟 路径变更: 从 core.loaders 导入
from core.loaders.asap_loader import ASAPLoader

# ==========================================
# 路径配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# 1. 数据集路径 (TSV)
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_submissions", "training_set_rel3.tsv")

# 2. 元数据路径 (JSON)
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata", "asap_context.json")


def calculate_reward(final_state, ground_truth_score):
    """奖励函数 (保持不变)"""
    reviews = final_state.get("reviews", [])
    if not reviews:
        return -1.0

    last_reviews = reviews[-3:]
    if not last_reviews:
        return 0.0

    avg_score = np.mean([r.overall_score for r in last_reviews])
    diff = abs(avg_score - ground_truth_score)

    accuracy_reward = 1.0 - diff
    rounds = final_state.get("current_round", 1)
    efficiency_penalty = 0.05 * (rounds - 1)

    total_reward = accuracy_reward - efficiency_penalty

    print(f"   🎯 Truth: {ground_truth_score:.2f} | Agents: {avg_score:.2f} | Diff: {diff:.2f}")
    print(f"   💰 Reward: {total_reward:.4f}")

    return total_reward


def train(episodes=10):
    print(f">>> 🚀 启动 ASAP 训练 (Episodes: {episodes})")
    print(f"    TSV: {os.path.basename(DATA_PATH)}")
    print(f"    JSON: {os.path.basename(METADATA_PATH)}")

    # 🌟 初始化 Loader (传入两个路径)
    try:
        loader = ASAPLoader(tsv_path=DATA_PATH, metadata_path=METADATA_PATH)
        loader.load_dataset()
    except FileNotFoundError as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 获取训练集索引
    train_indices = loader.get_split_indices('train')
    print(f"    训练集大小: {len(train_indices)}")

    global_dqn_agent.policy_net.train()

    for i in range(episodes):
        print(f"\n🎬 Episode {i + 1}/{episodes}")

        # 随机从训练集中采样
        rand_idx = np.random.choice(train_indices)
        subject, gt_score = loader.get_subject_by_index(rand_idx)

        print(f"   📝 ID: {subject.subject_id} (Set {subject.metadata['set_id']})")

        state = {
            "submission": subject,
            "reviews": [],
            "current_round": 1
        }

        try:
            final_state = mas_graph.invoke(
                state,
                config=RunnableConfig(run_name=f"Train_Ep_{i}")
            )

            reward = calculate_reward(final_state, gt_score)

            # 模拟梯度更新
            loss = global_dqn_agent.update_policy(batch_size=16)
            if loss:
                print(f"   🔥 Loss: {loss:.4f}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    global_dqn_agent.save(os.path.join(BASE_DIR, "core", "dqn_weights_asap.pth"))
    print("\n💾 模型已保存。")


if __name__ == "__main__":
    train(episodes=5)