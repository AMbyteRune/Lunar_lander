import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import imageio


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    train_env = make_vec_env("LunarLander-v3", n_envs=4)

    model = PPO("MlpPolicy", train_env, verbose=1, device="cpu")

    print("Starting baseline training (500k timesteps)...")
    model.learn(total_timesteps=500_000)

    model_path = "models/ppo_baseline"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Evaluate baseline model
    eval_env = Monitor(gym.make("LunarLander-v3"))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    eval_env.close()

    print(f"\nBaseline evaluation (10 episodes): {mean_reward:.2f} +/- {std_reward:.2f}")

    # Record evaluation video and GIF
    print("Recording baseline video and GIF...")
    video_env = gym.make("LunarLander-v3", render_mode="rgb_array")
    video_env = RecordVideo(
        video_env,
        video_folder="assets",
        name_prefix="ppo_baseline_eval",
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

    gif_path = "assets/ppo_baseline_eval.gif"
    imageio.mimsave(gif_path, frames, fps=50)
    print(f"Saved GIF to {gif_path}")

    # Test run with human rendering
    human_env = gym.make("LunarLander-v3", render_mode="human")
    obs, info = human_env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = human_env.step(action)
        if terminated or truncated:
            obs, info = human_env.reset()
    human_env.close()