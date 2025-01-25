import os
import gym
import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import RecordVideo
from env import pusher_v4

video_dir = "videos"

# Function for training
def train_model(env_id, total_timesteps, model_path, render_training=False):
    train_video = os.path.join(video_dir, "train")
    # Ensure the video directory exists
    os.makedirs(train_video, exist_ok=True)
    # Ensure the plot directory exists
    os.makedirs("plots", exist_ok=True)


    env = gym.make(env_id)
    eval_env = gym.make(env_id)  # Environment for evaluation

    # Configure the SAC model with optimized hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=int(1e6),
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
    )

    # Callback for periodic evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=20,
        deterministic=True,
    )

    rewards = []
    mean_rewards = []

    obs = env.reset()
    if render_training:

        for _ in range(total_timesteps):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            mean_rewards.append(sum(rewards) / len(rewards))
            env.render(mode='human')  # Render the environment
            if done:
                obs = env.reset()
    else:
        for _ in range(total_timesteps):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            mean_rewards.append(sum(rewards) / len(rewards))
            if done:
                obs = env.reset()
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save(model_path)
    print(f"Model saved at: {model_path}")
    env.close()

    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Reward')
    plt.plot(mean_rewards, label='Mean Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training Rewards')
    plt.savefig("plots/rewards.png")

# Function for testing
def test_model(env_id, model_path, n_episodes, video_dir):
    test_video = os.path.join(video_dir, "test")
    # Ensure the video directory exists
    os.makedirs(test_video, exist_ok=True)

    # Create the environment with render_mode="rgb_array"
    env = gym.make(env_id, render_mode="rgb_array", train=False)

    # Apply the RecordVideo wrapper
    env = RecordVideo(env, video_folder=test_video, episode_trigger=lambda x: True)

    # Load the SAC model
    model = SAC.load(model_path)

    # Run episodes to save the video
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()  # Render the environment
            if done:
                obs = env.reset()

    env.close()
    print(f"Video saved in: {test_video}")

# Main argument configuration
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Test SAC on Pusher")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Esegui 'train' o 'test'")
    parser.add_argument("--env", type=str, default="Pusher-v4", help="Gym environment ID")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of timesteps for training")
    parser.add_argument("--episodes", type=int, default=10, help="Numero di episodi per il test")
    parser.add_argument("--model-path", type=str, default="sac_pusher_OSTACOLI", help="Path to save the model")
    parser.add_argument("--render-training", action='store_true', help="Render the training process")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save videos")
    args = parser.parse_args()

    if args.mode == "train":
        train_model(args.env, args.timesteps, args.model_path, args.render_training)
    elif args.mode == "test":
        test_model(args.env, args.model_path, args.episodes, args.video_dir)
