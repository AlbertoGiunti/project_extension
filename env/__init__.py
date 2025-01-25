from gym.envs.registration import register

register(
    id="Pusher-v4",
    entry_point="env.pusher_v4:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)
