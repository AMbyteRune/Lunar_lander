import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import imageio


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_shaping = None

    def _get_shaping(self, obs):
        x, tilt, angular_vel = obs[0], obs[4], obs[5]
        # penalize distance from center, tilt angle, and angular velocity
        return -10.0 * abs(x) - 20.0 * abs(tilt) - 5.0 * abs(angular_vel)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_shaping = self._get_shaping(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # small step penalty to prevent hovering
        reward -= 0.05

        current_shaping = self._get_shaping(obs)
        if self.prev_shaping is not None:
            reward += (current_shaping - self.prev_shaping)
        self.prev_shaping = current_shaping

        # bonus for stable landing near the center pad
        left_leg, right_leg = obs[6], obs[7]
        if left_leg == 1.0 and right_leg == 1.0 and abs(obs[0]) < 0.2:
            reward += 0.2

        return obs, reward, terminated, truncated, info


def make_env():
    env = gym.make("LunarLander-v3")
    return CustomRewardWrapper(env)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    train_env = make_vec_env(make_env, n_envs=8)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        verbose=1,
        device="cpu"
    )

    print("Starting training for 1M timesteps...")
    model.learn(total_timesteps=1_000_000)

    model_path = "models/ppo_lunar_custom"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Evaluate on standard environment without the reward wrapper
    eval_env = Monitor(gym.make("LunarLander-v3"))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
    eval_env.close()

    print(f"Evaluation results (raw env): {mean_reward:.2f} +/- {std_reward:.2f}")

    # Record a test
    print("Recording evaluation video and GIF...")
    video_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    video_env = RecordVideo(
        video_env,
        video_folder="assets",
        name_prefix="ppo_landing_eval",
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

    gif_path = "assets/ppo_landing_eval.gif"
    imageio.mimsave(gif_path, frames, fps=50)
    print(f"Saved GIF to {gif_path}")

    # Test run with rendering
    human_env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = human_env.reset()
    for _ in range(600):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = human_env.step(action)
        if terminated or truncated:
            obs, info = human_env.reset()
    human_env.close()