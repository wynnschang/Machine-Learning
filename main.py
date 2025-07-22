import warnings

import numpy as np
import pandas as pd

from src.creditLimit_env import CreditLimitEnv
from src.dqn import DQNTrainer
from src.evaluation import EvaluationUtils

warnings.filterwarnings("ignore")


# === CONFIG ===
EPISODES = 200
N_RUNS = 20
MODEL_PATH = "models/best_dqn_model.pth"
DATA_PATH = "data/processed/cleaned_df.csv"

print("=== RL Pipeline Started ===")

# === Step 1: Load dataset ===
print("[INFO] Loading dataset...")
final_df = pd.read_csv(DATA_PATH)
print(f"[INFO] Loaded {len(final_df):,} rows.")

# === Step 2: Setup environment ===
provision_bins = np.arange(-0.5, 1.51, 0.01)
globals()["provision_bins"] = provision_bins
print("[INFO] Creating environments...")

# === Step 3: Train or load model ===
train_env = CreditLimitEnv(final_df.copy(), provision_bins=provision_bins)
trainer = DQNTrainer(train_env, model_path=MODEL_PATH)
model, _, reward_history, _ = trainer.train(episodes=EPISODES)
print(f"[INFO] Training reward history length: {len(reward_history)}")
EvaluationUtils.plot_training_reward_history(reward_history)

# === Step 4: Wrap model as RL agent ===
agent = EvaluationUtils.DQNAgent(model, train_env.action_space)

# === Step 5: Evaluate RL and benchmarks ===
print("[INFO] Evaluating RL agent and benchmarks...")

# Separate envs for evaluation
rl_env    = CreditLimitEnv(final_df.copy(), provision_bins=provision_bins)
rand_env  = CreditLimitEnv(final_df.copy(), provision_bins=provision_bins)
no_env    = CreditLimitEnv(final_df.copy(), provision_bins=provision_bins)
all_env   = CreditLimitEnv(final_df.copy(), provision_bins=provision_bins)

# Evaluate
rl_sim, rl_real     = EvaluationUtils.evaluate_policy(agent, rl_env, n_runs=N_RUNS, reward_history=reward_history, n_episodes=EPISODES)
rand_sim, rand_real = EvaluationUtils.evaluate_policy(None, rand_env, n_runs=N_RUNS, benchmark="random")
no_sim, no_real     = EvaluationUtils.evaluate_policy(None, no_env, n_runs=N_RUNS, benchmark="never_increase")
all_sim, all_real   = EvaluationUtils.evaluate_policy(None, all_env, n_runs=N_RUNS, benchmark="always_increase")

# === Step 6: Package results ===
sim_results = {
    "RL Agent": rl_sim,
    "Random": rand_sim,
    "Never Increase": no_sim,
    "Always Increase": all_sim
}
real_results = {
    "RL Agent": rl_real,
    "Random": rand_real,
    "Never Increase": no_real,
    "Always Increase": all_real
}

# === Step 7: Print summaries ===
print("\n=== Simulation Reward Summary ===")
for name, rewards in sim_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

print("\n=== Real Reward Summary ===")
for name, rewards in real_results.items():
    print(f"{name:17s}: {np.mean(rewards):.4f}")

# === Step 8: Plot comparisons ===
EvaluationUtils.plot_policy_comparison(sim_results, real_results)

print("=== RL Pipeline Completed ===")