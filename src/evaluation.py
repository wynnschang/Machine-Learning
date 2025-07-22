import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from collections import defaultdict


DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"
if not DEBUG_MODE:
    os.makedirs("results", exist_ok=True)

class EvaluationUtils:

    class DQNAgent:
        def __init__(self, model, action_space):
            self.model = model
            self.device = next(model.parameters()).device
            self.action_space = action_space

        def choose_action(self, state):
            state_array = np.asarray(state, dtype=np.float32).reshape(1, -1)
            state_tensor = torch.from_numpy(state_array).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    @staticmethod
    def evaluate_policy(agent, env, n_runs=10, benchmark=None, reward_history=None, n_episodes=None):
        synthetic_rewards = []
        real_rewards = []
        action_dist = defaultdict(lambda: {0: 0, 1: 0})
        real_rewards_by_class = defaultdict(list)
        provision_changes = []
        total_customers = 0

        # Try loading reward history from file if not provided
        if reward_history is None and os.path.exists("results/training_reward_history.csv"):
            reward_history = pd.read_csv("results/training_reward_history.csv")["reward"].tolist()
            if DEBUG_MODE:
                print(f"[DEBUG] Loaded reward history from CSV with {len(reward_history)} episodes.")

        # Determine episode count for reporting
        if reward_history is not None:
            episode_count = len(reward_history)
        elif n_episodes is not None:
            episode_count = n_episodes
        elif os.path.exists("models/best_dqn_model_meta.json"):
            try:
                import json
                with open("models/best_dqn_model_meta.json", "r") as f:
                    episode_count = json.load(f).get("episodes", "N/A")
            except:
                episode_count = "N/A"
        else:
            episode_count = "N/A"

        for run in range(n_runs):
            state = env.reset()
            done = False
            total_syn = 0.0
            total_real = 0.0

            while not done:
                if benchmark == "always_increase":
                    action = 1
                elif benchmark == "never_increase":
                    action = 0
                elif benchmark == "random":
                    action = np.random.choice(env.action_space)
                else:
                    action = agent.choose_action(state)

                next_state, reward, done, info = env.step(action)
                total_syn += float(reward)
                total_customers += 1

                cls = info.get("BALANCE_CLASS", -1)
                if cls in [0, 1]:
                    action_dist[cls][action] += 1

                bal = float(info.get("actual_balance", 0))
                rate = float(info.get("interest_rate", 0))
                pd_ = float(info.get("pd", 0.05))
                lgd = float(info.get("lgd", 0.5))
                limit = float(info.get("new_limit", 0))
                ccf = float(info.get("ccf", 0.8))
                delta_prov = float(info.get("delta_provision", 0))
                provision_changes.append(delta_prov)

                ead = bal + ccf * (limit - bal)
                real_reward = 3 * rate * bal * (1 - pd_) - pd_ * lgd * ead
                real_reward = np.clip(real_reward, -1e6, 1e6) / 1e6
                total_real += real_reward

                if cls in [0, 1]:
                    real_rewards_by_class[cls].append(real_reward)

                state = next_state

            synthetic_rewards.append(total_syn)
            real_rewards.append(total_real)

        is_rl_policy = (benchmark is None and agent is not None)

        if not DEBUG_MODE and is_rl_policy:
            with open("results/DQN_results.txt", "w", encoding="utf-8") as f:
                f.write("Evaluation Summary\n")
                f.write("===================\n")
                f.write(f"Total Episodes (DQN Training): {episode_count}\n")
                f.write(f"Evaluation Runs: {n_runs}\n")
                f.write(f"Total Customers Evaluated: {total_customers}\n\n")

                f.write("Action Selection Distribution:\n")
                for cls in [0, 1]:
                    total = action_dist[cls][0] + action_dist[cls][1]
                    if total > 0:
                        pct_0 = 100 * action_dist[cls][0] / total
                        pct_1 = 100 * action_dist[cls][1] / total
                    else:
                        pct_0 = pct_1 = 0.0
                    f.write(
                        f"  Class {cls} — No Increase: {action_dist[cls][0]} ({pct_0:.1f}%), "
                        f"Increase: {action_dist[cls][1]} ({pct_1:.1f}%)\n"
                    )
                f.write("\n")

                f.write("Average Real Reward by Class:\n")
                for cls in [0, 1]:
                    avg_reward = np.mean(real_rewards_by_class[cls]) if real_rewards_by_class[cls] else 0.0
                    f.write(f"  Class {cls}: {avg_reward:.4f}\n")
                f.write("\n")

                f.write("Provision Impact:\n")
                f.write(f"  Average ΔProvision: {np.mean(provision_changes):.4f}\n")
                f.write(f"  Max ΔProvision: {np.max(provision_changes):.4f}\n\n")

                f.write("Best Episode Reward:\n")
                f.write(f"  Simulated: {max(synthetic_rewards):.4f}\n")
                f.write(f"  Real     : {max(real_rewards):.4f}\n\n")

        return synthetic_rewards, real_rewards

    @staticmethod
    def plot_policy_comparison(results_dict_sim, results_dict_real):
        labels = list(results_dict_sim.keys())
        x = np.arange(len(labels))
        width = 0.35

        means_sim = [np.mean(results_dict_sim[k]) for k in labels]
        std_sim = [np.std(results_dict_sim[k]) / np.sqrt(len(results_dict_sim[k])) for k in labels]

        means_real = [np.mean(results_dict_real[k]) for k in labels]
        std_real = [np.std(results_dict_real[k]) / np.sqrt(len(results_dict_real[k])) for k in labels]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, means_sim, width, yerr=std_sim, label='Simulated', capsize=4)
        ax.bar(x + width / 2, means_real, width, yerr=std_real, label='Real', capsize=4)
        ax.set_ylabel("Average Total Reward")
        ax.set_title("Policy Evaluation: Simulated vs Real Rewards")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True)

        # Add numeric labels on bars
        for i, label in enumerate(labels):
            ax.text(x[i] - width / 2, means_sim[i], f"{means_sim[i]:.2f}", ha='center', va='bottom', fontsize=8)
            ax.text(x[i] + width / 2, means_real[i], f"{means_real[i]:.2f}", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        if not DEBUG_MODE:
            plt.savefig("results/policy_comparison.png")
            with open("results/DQN_results.txt", "a", encoding="utf-8") as f:
                f.write("Policy Comparison (Simulated vs Real Rewards)\n")
                f.write("-------------------------------------------------\n")
                for i, label in enumerate(labels):
                    f.write(f"{label}:\n")
                    f.write(f"  Simulated: {means_sim[i]:.4f} ± {std_sim[i]:.4f}\n")
                    f.write(f"  Real     : {means_real[i]:.4f} ± {std_real[i]:.4f}\n\n")

        plt.show()

    @staticmethod
    def plot_training_reward_history(reward_history):
        if reward_history is None or len(reward_history) == 0:
            print("[WARNING] No reward history available to plot.")
            return

        reward_array = np.asarray(reward_history, dtype=np.float32)
        if reward_array.ndim != 1:
            print(f"[ERROR] reward_history must be 1D, got shape: {reward_array.shape}")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(reward_array, linewidth=1.5)  # Removed marker='o'
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Reward Curve")
        plt.grid(True)
        plt.tight_layout()

        if not DEBUG_MODE:
            plt.savefig("results/training_reward.png")
            pd.DataFrame({"episode": np.arange(len(reward_array)), "reward": reward_array}).to_csv(
                "results/training_reward_history.csv", index=False
            )

        plt.show()