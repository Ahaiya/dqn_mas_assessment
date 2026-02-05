"""
Final Training Script (Configuration Driven)
============================================
"""
import os
import sys
import numpy as np
import pandas as pd
import math
import torch
from datetime import datetime
from langchain_core.runnables import RunnableConfig

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflow.graph import mas_graph
from workflow.dqn_node import global_dqn_agent
from core.loaders.asap_loader import ASAPLoader
from config.loader import global_config

# 1. 读取配置
CONF = global_config["training"]
RUN_MODE = global_config.get("run_mode", "unknown")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_submissions", "training_set_rel3.tsv")
METADATA_PATH = os.path.join(BASE_DIR, "data", "metadata", "asap_context.json")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 定义存档路径 (用于断点续训)
CHECKPOINT_PATH = os.path.join(BASE_DIR, "data", "model", "dqn_checkpoint.pth")

def get_epsilon(episode_idx):
    """计算当前轮次的探索率 (指数衰减)"""
    start = CONF["epsilon_start"]
    end = CONF["epsilon_end"]
    decay = CONF["epsilon_decay"]
    return end + (start - end) * math.exp(-1. * episode_idx / decay)


def calculate_reward(final_state, ground_truth_score):
    """
    奖励函数设计: Accuracy (准确性) - Efficiency (效率)
    """
    reviews = final_state.get("reviews", [])
    if not reviews:
        return -1.0, 0.0

    # 1. 计算预测分 (取最后3个专家的平均值，假设是3个专家)
    num_agents = 3
    last_reviews = reviews[-num_agents:]
    pred_score = np.mean([r.overall_score for r in last_reviews])

    # 2. 计算误差 (0-5分制)
    error = abs(pred_score - ground_truth_score)
    # 3. 准确性奖励 (满分 1.0)
    ## 奖励: 误差越小越好。满分 1.0。
    # 策略: error=0 -> 1.0; error=1 -> 0.6; error>=2.5 -> 0
    acc_reward = max(0, 1.0 - (error * 0.4))

    # 4. 效率惩罚 (每多一轮扣 0.05)
    rounds = final_state.get("current_round", 1)
    actual_rounds = max(1, rounds - 1)

    eff_penalty = 0.05 * (actual_rounds - 1)

    total = acc_reward - eff_penalty
    return total, pred_score

def save_checkpoint(episode, agent):
    """保存完整训练状态"""
    state = {
        'episode': episode,
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        # 如果需要，这里也可以存 target_net，但通常 load 时重新 sync 即可
    }
    torch.save(state, CHECKPOINT_PATH)
    print(f" Checkpoint saved to {CHECKPOINT_PATH}")

def load_checkpoint(agent):
    """加载断点"""
    if not os.path.exists(CHECKPOINT_PATH):
        print(" No checkpoint found, starting from scratch.")
        return 0

    try:
        checkpoint = torch.load(CHECKPOINT_PATH)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        agent.target_net.load_state_dict(checkpoint['model_state_dict']) # 同步 Target
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        print(f" Resuming from Episode {start_episode}")
        return start_episode
    except Exception as e:
        print(f" Checkpoint load failed ({e}), starting from scratch.")
        return 0


def train():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n>>>  Starting DQN Training | Mode: {RUN_MODE} | Session: {timestamp}")
    print(f"    Params: Ep={CONF['total_episodes']}, Batch={CONF['batch_size']}, LR={CONF['learning_rate']}")

    # 1. 加载数据
    loader = ASAPLoader(tsv_path=DATA_PATH, metadata_path=METADATA_PATH)
    try:
        loader.load_dataset()
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return

    train_indices = loader.get_split_indices('train')
    start_episode = load_checkpoint(global_dqn_agent)  # 🌟 加载断点

    # 2. 训练循环
    global_dqn_agent.policy_net.train()
    metrics_log = []

    for i in range(start_episode, CONF["total_episodes"]):
        epsilon = get_epsilon(i)

        # A. 随机采样一个样本 (Essay)
        idx = np.random.choice(train_indices)
        subject, gt_score = loader.get_subject_by_index(idx)

        # B. 初始化图状态
        state = {
            "submission": subject,
            "reviews": [],
            "current_round": 1,
            "epsilon": epsilon,
            "dqn_trace": [],    # 轨迹容器
            "dqn_action": -1
        }

        try:
            # C. 运行 Graph
            final_state = mas_graph.invoke(state, config=RunnableConfig(run_name=f"Ep_{i}"))

            # D. 结算奖励
            reward, pred_score = calculate_reward(final_state, gt_score)

            # E. 存储经验 (Hindsight Experience Replay)
            trace = final_state.get("dqn_trace", [])

            if trace:
                for t, (s, a) in enumerate(trace):
                    is_last = (t == len(trace) - 1)
                    # Next State:
                    # 如果不是最后一步，next_s 就是 trace[t+1][0] (即下一轮的状态)
                    # 如果是最后一步，next_s 无意义 (因为 done=True)，填当前 s 保持格式
                    next_s = trace[t + 1][0] if not is_last else s

                    # Reward: 稀疏奖励，只在最后一步给
                    step_r = reward if is_last else 0.0

                    # 存入 Buffer
                    global_dqn_agent.store_transition(s, a, step_r, next_s, is_last)

            # 更新网络 (仅在 Buffer 足够且过预热期后)
            loss = None
            # if i > CONF["warmup_steps"]:
            #     loss = global_dqn_agent.update_policy(batch_size=CONF["batch_size"])
            loss = global_dqn_agent.update_policy(batch_size=CONF["batch_size"])

            # 打印日志 (每 10 轮)
            if (i + 1) % 10 == 0:
                print(f"Ep {i + 1:04d} | Eps: {epsilon:.2f} | Rds: {final_state['current_round']} | "
                      f"GT: {gt_score:.1f} vs Pred: {pred_score:.1f} | Rw: {reward:.3f} | Loss: {loss}")
                # 🌟 定期保存 Checkpoint
                save_checkpoint(i, global_dqn_agent)

            metrics_log.append({
                "episode": i,
                "reward": reward,
                "loss": loss,
                "rounds": final_state['current_round'] - 1,
                "epsilon": epsilon,
                "gt": gt_score,
                "pred": pred_score
            })

        except Exception as e:
            print(f"❌ Ep {i} Runtime Error: {e}")
            import traceback
            traceback.print_exc()

    # 3. 结束保存
    log_path = os.path.join(LOG_DIR, f"train_log_{timestamp}.csv")
    pd.DataFrame(metrics_log).to_csv(log_path, index=False)

    model_path = os.path.join(BASE_DIR, "data", "model", "dqn_weights_final.pth")
    global_dqn_agent.save(model_path)

    print(f"\n Training Finished. Log: {log_path}, Model: {model_path}")


if __name__ == "__main__":
    train()