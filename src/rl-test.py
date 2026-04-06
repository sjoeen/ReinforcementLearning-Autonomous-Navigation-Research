import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from wheelchair_env import WheelchairEnv
from stable_baselines3.common.monitor import Monitor
import csv

TIME_STEPS = 60_000
N_ROBOTS = 9

def run_model():
    """Start vectorized environment to test model in parallel"""
    def env_fn(i):
        def _init():
            return Monitor(WheelchairEnv(i))
        return _init

    env = SubprocVecEnv([env_fn(i) for i in range(N_ROBOTS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # CHANGE THIS PATH TO THE MODEL YOU WANT TO TEST BEFORE RUNNING!
    path = "./models/ppo-good" 
    assert os.path.exists(path + ".zip"), "Model path does not exist. Please train the model first."

    model = PPO.load(path, env)

    success_counts = [0] * N_ROBOTS
    episode_counts = [0] * N_ROBOTS
    total_times = [0.0] * N_ROBOTS
    total_distances = [0.0] * N_ROBOTS
    
    start_times = [time.time()] * N_ROBOTS
    episode_distances = [[] for _ in range(N_ROBOTS)]

    print("Testing started! Please wait...")
    obs = env.reset()
    
    for _ in range(TIME_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        for i in range(N_ROBOTS):
            # Guardar a distância mínima lida pelo LiDAR neste instante
            episode_distances[i].append(np.min(obs[i]))

            if dones[i]:
                episode_counts[i] += 1
                if infos[i].get("is_success", False):
                    success_counts[i] += 1
                
                # Calcular o tempo que o episódio demorou
                end_time = time.time()
                total_times[i] += (end_time - start_times[i])
                start_times[i] = time.time() # Reset ao cronómetro
                
                # Calcular a distância média ao longo deste episódio
                total_distances[i] += np.mean(episode_distances[i])
                episode_distances[i] = [] # Reset às distâncias

    print("\n=== FINAL RESULTS ===")
    with open("results_evaluation.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["robot_id", "success_rate", "avg_time", "avg_distance"])
        
        for i in range(N_ROBOTS):
            if episode_counts[i] > 0:
                rate = 100 * success_counts[i] / episode_counts[i]
                avg_t = total_times[i] / episode_counts[i]
                avg_d = total_distances[i] / episode_counts[i]
                print(f"Robot {i}: {rate:.1f}% Success | Time: {avg_t:.1f}s | Dist: {avg_d:.2f}")
            else:
                rate = avg_t = avg_d = 0
                print(f"Robot {i}: no completed episodes.")
                
            writer.writerow([i, f"{rate:.2f}", f"{avg_t:.2f}", f"{avg_d:.2f}"])
            
    print("\nResults saved to 'results_evaluation.csv'")

if __name__ == "__main__":
    run_model()
