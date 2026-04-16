import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from wheelchair_env import WheelchairEnv
from stable_baselines3.common.monitor import Monitor
import csv

TIME_STEPS = 60_000
N_ROBOTS = 3
OUTPUT_FILE = "results_evaluation.csv"
AGG_FILE = "results_aggregate.csv"

def run_model():
    def env_fn(i):
        def _init():
            return Monitor(WheelchairEnv(i))
        return _init

    env = SubprocVecEnv([env_fn(i) for i in range(N_ROBOTS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # CHANGE THIS PATH BEFORE RUNNING
    path = "models/model_A_baseline"
    assert os.path.exists(path + ".zip"), "Model path does not exist."

    model = PPO.load(path, env)

    # ── Counters ─────────────────────────────────────────────
    success_counts   = [0] * N_ROBOTS
    collision_counts = [0] * N_ROBOTS
    timeout_counts   = [0] * N_ROBOTS
    episode_counts   = [0] * N_ROBOTS

    total_times        = [0.0] * N_ROBOTS
    total_steps        = [0]   * N_ROBOTS
    total_mean_dist    = [0.0] * N_ROBOTS
    total_min_dist     = [0.0] * N_ROBOTS

    start_times        = [time.time()] * N_ROBOTS
    episode_steps      = [0] * N_ROBOTS
    episode_distances  = [[] for _ in range(N_ROBOTS)]

    print("Testing started! Please wait...")
    obs = env.reset()

    for _ in range(TIME_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        for i in range(N_ROBOTS):
            episode_steps[i] += 1
            episode_distances[i].append(np.min(obs[i]))

            if dones[i]:
                episode_counts[i] += 1

                # ── Outcome classification ───────────────────
                is_success = infos[i].get("is_success", False)
                is_collision = (not is_success) and (episode_steps[i] < 19000)
                is_timeout   = (not is_success) and (episode_steps[i] >= 19000)

                if is_success:
                    success_counts[i] += 1
                elif is_collision:
                    collision_counts[i] += 1
                else:
                    timeout_counts[i] += 1

                # ── Time ─────────────────────────────────────
                end_time = time.time()
                total_times[i] += (end_time - start_times[i])
                start_times[i] = time.time()

                # ── Steps ────────────────────────────────────
                total_steps[i] += episode_steps[i]

                # ── Distance metrics ─────────────────────────
                if episode_distances[i]:
                    ep_mean = np.mean(episode_distances[i])
                    ep_min  = np.min(episode_distances[i])
                else:
                    ep_mean = ep_min = 0.0

                total_mean_dist[i] += ep_mean
                total_min_dist[i]  += ep_min

                # ── Reset episode ────────────────────────────
                episode_steps[i] = 0
                episode_distances[i] = []

    print("\n=== FINAL RESULTS ===")

    # ── Per-robot CSV ─────────────────────────────────────────
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "robot_id", "episodes",
            "success_rate_%", "collision_rate_%", "timeout_rate_%",
            "avg_steps", "avg_time_s",
            "avg_mean_distance", "avg_min_distance"
        ])

        for i in range(N_ROBOTS):
            n = max(episode_counts[i], 1)

            sr = 100 * success_counts[i]   / n
            cr = 100 * collision_counts[i] / n
            tr = 100 * timeout_counts[i]   / n
            avg_steps = total_steps[i]     / n
            avg_time  = total_times[i]     / n
            avg_mean  = total_mean_dist[i] / n
            avg_min   = total_min_dist[i]  / n

            print(f"Robot {i}: {sr:.1f}% success | {cr:.1f}% collision | {tr:.1f}% timeout | "
                  f"{avg_steps:.0f} steps | mean_d={avg_mean:.2f} | min_d={avg_min:.2f}")

            writer.writerow([
                i, episode_counts[i],
                f"{sr:.2f}", f"{cr:.2f}", f"{tr:.2f}",
                f"{avg_steps:.1f}", f"{avg_time:.2f}",
                f"{avg_mean:.4f}", f"{avg_min:.4f}"
            ])

    print(f"\nSaved: {OUTPUT_FILE}")

    # ── Aggregate CSV ─────────────────────────────────────────
    total_ep = sum(episode_counts)
    n = max(total_ep, 1)

    with open(AGG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "episodes",
            "success_rate_%", "collision_rate_%", "timeout_rate_%",
            "avg_steps", "avg_mean_distance", "avg_min_distance"
        ])

        writer.writerow([
            os.path.basename(path), total_ep,
            f"{100*sum(success_counts)/n:.2f}",
            f"{100*sum(collision_counts)/n:.2f}",
            f"{100*sum(timeout_counts)/n:.2f}",
            f"{sum(total_steps)/n:.1f}",
            f"{sum(total_mean_dist)/n:.4f}",
            f"{sum(total_min_dist)/n:.4f}"
        ])

    print(f"Saved: {AGG_FILE}")


if __name__ == "__main__":
    run_model()