import os
import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import RecordVideo
from env import pusher_v4

video_dir = "videos"


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_rewards(data, plot_dir):
    """
    Generates a plot of episodic rewards and saves it to the specified directory.
    """
    timesteps = data['timesteps']
    rewards = data['results']

    # Calculate the mean reward for each timestep across the episodes
    mean_rewards_per_timestep = rewards.mean(axis=1)

    # Calculate the moving average of the mean rewards with a window size of 50
    window_size = 50
    mean_rewards = moving_average(mean_rewards_per_timestep, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards_per_timestep, label="Reward per timestep", color="green", alpha=0.5)
    plt.plot(timesteps[window_size-1:], mean_rewards, label="Mean reward", color='blue')
    plt.xlabel("Number of timesteps")
    plt.ylabel("Reward")
    plt.title("Reward during training")
    plt.legend()
    plt.grid()

    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "rewards_plot.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Reward plot saved in {plot_path}")


# Function for training
def train_model(env_id, total_timesteps, model_path, render_training=False):
    train_video = os.path.join(video_dir, "train")
    # Ensure the video directory exists
    os.makedirs(train_video, exist_ok=True)
    # Ensure the plot directory exists
    os.makedirs("plots", exist_ok=True)

    env = gym.make(env_id, obstacle_random=args.obstacle_random, udr=args.udr)  # Environment for training
    eval_env = gym.make(env_id, obstacle_random=args.obstacle_random, udr=args.udr)  # Environment for evaluation

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
    reward_logger = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=100,
        deterministic=True,
    )


    obs = env.reset()
    if render_training:

        for _ in range(total_timesteps):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render(mode='human')  # Render the environment
            if done:
                obs = env.reset()
    else:
        for _ in range(total_timesteps):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        model.learn(total_timesteps=total_timesteps, callback=reward_logger)

    # Save the model
    model.save(model_path)
    print(f"Model saved at: {model_path}")
    env.close()

    # Salva i log
    if reward_logger.evaluations_results is not None:
        evals = reward_logger.evaluations_results
        timesteps = reward_logger.evaluations_timesteps
        np.savez(os.path.join("logs", "evaluations.npz"), timesteps=timesteps, results=evals)

    # Plot the rewards

# Function for testing
def test_model(env_id, model_path, n_episodes, video_dir, render_test=False):
    test_video = os.path.join(video_dir, "test")
    # Ensure the video directory exists
    os.makedirs(test_video, exist_ok=True)

    # Create the environment with render_mode="rgb_array"
    env = gym.make(env_id, render_mode="rgb_array", train=False, obstacle_random=args.obstacle_random, udr=args.udr)

    # Apply the RecordVideo wrapper
    #env = RecordVideo(env, video_folder=test_video, episode_trigger=lambda x: True)

    # Load the SAC model
    model = SAC.load(model_path)

    # Run episodes to save the video
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            #env.render()  # Render the environment
            if render_test:
                env.render()  # Render the environment
            if done:
                obs = env.reset()

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes, render=True)
    print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

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
    parser.add_argument("--render-test", action='store_true', help="Render the test process")
    parser.add_argument("--video-dir", type=str, default="videos", help="Directory to save videos")
    parser.add_argument("--obstacle-random", action='store_true', help="Random object position")
    parser.add_argument("--udr", action='store_true', help="Uniform domain randomization del peso dell'avambraccio")
    args = parser.parse_args()

    if args.mode == "train":
        evaluation_file = os.path.join("logs", "evaluations.npz")
        train_model(args.env, args.timesteps, args.model_path, args.render_training)
        if os.path.exists(evaluation_file):
            data = np.load(evaluation_file)
            plot_rewards(data, "plots")
    elif args.mode == "test":
        test_model(args.env, args.model_path, args.episodes, args.video_dir, args.render_test)


'''
TRAIN
python pusher_obstacles.py --mode train --env Pusher-v4 --timesteps 500000 --model-path sac_pusher_Obs3_random_500k --object-random 

TEST
 python pusher_obstacles.py --mode test --env Pusher-v4 --model-path sac_pusher_Obs3_random_500k --episodes 10 --video-dir videos --object-random
'''
