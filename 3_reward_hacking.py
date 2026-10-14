import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import imageio


class HoverRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        vx, vy = obs[2], obs[3]
        tilt = obs[4]
        left_leg, right_leg = obs[6], obs[7]

        in_air = (left_leg == 0) and (right_leg == 0)
        is_upright = abs(tilt) < 0.2
        is_slow = (abs(vx) < 0.3) and (abs(vy) < 0.3)

        # flawed reward logic: +1 bonus per frame for hovering
        if in_air and is_upright and is_slow:
            reward += 1.0

        return obs, reward, terminated, truncated, info


def make_hacked_env():
    env = gym.make("LunarLander-v3")
    return HoverRewardWrapper(env)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    train_env = make_vec_env(make_hacked_env, n_envs=4)

    model = PPO("MlpPolicy", train_env, verbose=1, device="cpu")

    print("Starting training with flawed reward wrapper (500k timesteps)...")
    model.learn(total_timesteps=500_000)

    model_path = "models/ppo_reward_hacked"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Evaluate on standard environment (true performance)
    real_eval_env = Monitor(gym.make("LunarLander-v3"))
    real_mean, real_std = evaluate_policy(model, real_eval_env, n_eval_episodes=10, deterministic=True)
    real_eval_env.close()

    # Evaluate on wrapped environment (exploited reward)
    hacked_eval_env = Monitor(make_hacked_env())
    hacked_mean, _ = evaluate_policy(model, hacked_eval_env, n_eval_episodes=10, deterministic=True)
    hacked_eval_env.close()

    print("\n--- Evaluation Results ---")
    print(f"Wrapped env mean reward: {hacked_mean:.2f}")
    print(f"Standard env mean reward: {real_mean:.2f}")

    # Record evaluation video and GIF
    print("\nRecording video and GIF...")
    video_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    video_env = RecordVideo(
        video_env,
        video_folder="assets",
        name_prefix="ppo_hover_exploit",
        episode_trigger=lambda x: True
    )

    frames = []
    obs, info = video_env.reset()
    done = False

    while not done:
        frames.append(video_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = video_env.step(action)
        done = terminated or truncated

    video_env.close()

    gif_path = "assets/ppo_hover_exploit.gif"
    imageio.mimsave(gif_path, frames, fps=50)
    print(f"Saved GIF to {gif_path}")

    # Test run with human rendering
    human_env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = human_env.reset()
    for _ in range(800):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = human_env.step(action)
        if terminated or truncated:
            obs, info = human_env.reset()
    human_env.close()