import argparse
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import RecordVideo
import os
from env.custom_pusher import *




video_dir = "videos"

# Funzione per il training
def train_model(env_id, total_timesteps, model_path):
    env = gym.make(env_id)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"Modello salvato in: {model_path}")
    env.close()


# Funzione per il test con registrazione video
def test_model(env_id, model_path, n_episodes, video_dir):
    # Assicurati che la directory per i video esista
    os.makedirs(video_dir, exist_ok=True)

    # Crea l'ambiente senza render_mode
    env = gym.make(env_id)

    # Applica il wrapper RecordVideo
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True)

    # Carica il modello PPO
    model = PPO.load(model_path)

    # Valuta il modello
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    print(f"Ricompensa media: {mean_reward:.2f} ± {std_reward:.2f}")

    # Esegui un episodio per salvare il video
    obs = env.reset()
    for _ in range(1000):  # Esegui un episodio lungo
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
    print(f"Video salvato in: {video_dir}")


# Configura l'argomento principale
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test PPO on Pusher")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Esegui 'train' o 'test'")
    parser.add_argument("--env", type=str, default="CustomPusher-v0", help="ID dell'ambiente Gymnasium")
    parser.add_argument("--timesteps", type=int, default=100000, help="Numero di timesteps per il training")
    parser.add_argument("--model-path", type=str, default="ppo_pusher", help="Percorso per salvare/caricare il modello")
    parser.add_argument("--episodes", type=int, default=10, help="Numero di episodi per il test")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory per salvare i video")
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.env, args.timesteps, args.model_path)
    elif args.mode == "test":
        test_model(args.env, args.model_path, args.episodes, args.video_dir)

'''
Comando per eseguire quello di mujoco:
 python pusher_rl.py --mode test --model-path def_pusher_10M --episodes 8 --video-dir videos --env Pusher-v2


'''
