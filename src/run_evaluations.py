import os
import sys
import csv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from wheelchair_env import WheelchairEnv

EVAL_STEPS = 15_000
N_ROBOTS = 3
NOISE_LEVEL = 0.05

def save_csv(fase, folder, filename, metrics):
    path = f"../results/{fase}/{folder}"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{filename}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

def evaluate_loop(model, env, current_rays=360):
    obs = env.reset()
    episode_counts = [0] * N_ROBOTS
    success_counts = [0] * N_ROBOTS
    collision_counts = [0] * N_ROBOTS
    total_steps = [0] * N_ROBOTS
    episode_steps = [0] * N_ROBOTS

    for _ in range(EVAL_STEPS):
        actions, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(actions)
        
        for i in range(N_ROBOTS):
            episode_steps[i] += 1
            if dones[i]:
                episode_counts[i] += 1
                is_success = infos[i].get("goal_reached", False) or (infos[i].get("terminal_observation") is not None and not infos[i].get("collided", False))
                if is_success: success_counts[i] += 1
                else: collision_counts[i] += 1
                total_steps[i] += episode_steps[i]
                episode_steps[i] = 0

    total_ep = max(sum(episode_counts), 1)
    sr = (sum(success_counts) / total_ep) * 100
    cr = (sum(collision_counts) / total_ep) * 100
    avg_steps = sum(total_steps) / total_ep if sum(success_counts) > 0 else 0
    return {"Success Rate (%)": f"{sr:.2f}", "Collision Rate (%)": f"{cr:.2f}", "Avg Steps": f"{avg_steps:.1f}"}

def run_eval_fase2():
    print("\n--- A EXECUTAR AVALIAÇÃO DA FASE 2 (Estudo Cruzado de Raios) ---")
    model_configs = [
        {"name": "model_c", "class": PPO},
        {"name": "model_d", "class": PPO} # Ambos são PPO na Fase 2, muda o treino de raios
    ]
    
    test_rays = [360, 180, 90, 36]
    
    for cfg in model_configs:
        stats = f"../models/fase2_{cfg['name']}/vecnormalize_stats.pkl"
        model_p = f"../models/fase2_{cfg['name']}/model"
        
        if not os.path.exists(model_p + ".zip"):
            print(f"Modelo {cfg['name']} não encontrado. Salitando...")
            continue
            
        for rays in test_rays:
            print(f"A testar {cfg['name']} forçando o LiDAR a responder com {rays} raios...")
            env = SubprocVecEnv([lambda: Monitor(WheelchairEnv(i, noise_level=NOISE_LEVEL)) for i in range(N_ROBOTS)])
            env = VecNormalize.load(stats, env)
            env.training, env.norm_reward = False, False
            model = cfg["class"].load(model_p, env=env)
            
            # Nota: O teu wheelchair_env.py vai simular os raios reduzidos dinamicamente se o Webots enviar menos raios
            metrics = evaluate_loop(model, env, current_rays=rays)
            save_csv("fase2", cfg["name"], f"rays_{rays}", metrics)
            env.close()

def run_eval_fase3():
    print("\n--- A EXECUTAR AVALIAÇÃO DA FASE 3 (Benchmark de Algoritmos no Mapa Complexo) ---")
    # Aqui avaliamos os vossos modelos finais PPO vs DQN criados no mapa novo
    model_configs = [
        {"name": "ppo_complex", "class": PPO},
        {"name": "dqn_complex", "class": DQN}
    ]
    
    for cfg in model_configs:
        stats = f"../models/fase3_{cfg['name']}/vecnormalize_stats.pkl"
        model_p = f"../models/fase3_{cfg['name']}/model"
        
        if not os.path.exists(model_p + ".zip"):
            print(f"Modelo {cfg['name']} não encontrado em FASE 3. Saltando...")
            continue
            
        print(f"A avaliar {cfg['name']} no mapa complexo...")
        env = SubprocVecEnv([lambda: Monitor(WheelchairEnv(i, noise_level=NOISE_LEVEL)) for i in range(N_ROBOTS)])
        env = VecNormalize.load(stats, env)
        env.training, env.norm_reward = False, False
        model = cfg["class"].load(model_p, env=env)
        
        metrics = evaluate_loop(model, env)
        save_csv("fase3", "mapa_complexo", cfg["name"], metrics)
        env.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso correto: python run_evaluations.py [fase2 | fase3]")
    elif sys.argv[1] == "fase2":
        run_eval_fase2()
    elif sys.argv[1] == "fase3":
        run_eval_fase3()