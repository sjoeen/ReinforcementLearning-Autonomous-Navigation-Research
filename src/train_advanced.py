import os
import sys
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO, DQN
from wheelchair_env import WheelchairEnv
from cnn_feature_extractor import LidarCNNFeatureExtractor

N_ROBOTS = 3
NOISE_LEVEL = 0.05

def make_env(env_id, noise):
    return lambda: Monitor(WheelchairEnv(env_id, noise_level=noise))

def run_training(fase, model_name):
    fase = fase.lower()
    model_name = model_name.upper()
    
    print(f"\n>>> INICIANDO TREINO: {model_name} na {fase.upper()} <<<")
    
    # Inicializar ambientes
    env = SubprocVecEnv([make_env(i, NOISE_LEVEL) for i in range(N_ROBOTS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Organização de pastas por Fase para não misturar nada
    model_dir = f"../models/{fase}_{model_name.lower()}"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model")
    stats_path = os.path.join(model_dir, "vecnormalize_stats.pkl")

    policy_kwargs = dict(
        features_extractor_class=LidarCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # Escolha do algoritmo baseado no nome do modelo ou fase
    if "DQN" in model_name or model_name == "MODEL_D_DQN":
        model = DQN(
            "CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
            learning_rate=1e-4, buffer_size=50000, learning_starts=1000,
            batch_size=256, target_update_interval=1000, device="cpu",
            tensorboard_log=f"../logs/{fase}_{model_name.lower()}"
        )
        steps = 800_000  # DQN pode precisar de ajustes de passos
    else:
        model = PPO(
            "CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
            n_steps=4096, learning_rate=5e-5, batch_size=512, n_epochs=20,
            clip_range=0.1, ent_coef=0.01, device="cpu",
            tensorboard_log=f"../logs/{fase}_{model_name.lower()}"
        )
        steps = 1_000_000

    try:
        model.learn(total_timesteps=steps)
    except KeyboardInterrupt:
        print("\nTreino interrompido. A guardar progresso...")
    finally:
        model.save(model_path)
        env.save(stats_path)
        env.close()
        print(f"Sucesso! Guardado em: {model_path}.zip")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso correto: python train_advanced.py [fase2|fase3] [nome_modelo]")
        print("Exemplos:\n  python train_advanced.py fase2 model_C\n  python train_advanced.py fase3 ppo_complex")
    else:
        run_training(sys.argv[1], sys.argv[2])