import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="human")

    obs, info = env.reset()

    print("\n--- Initial Observation Vector ---")
    print(f"1. Horizontal pos (X):    {obs[0]:.2f}")
    print(f"2. Vertical pos (Y):      {obs[1]:.2f}")
    print(f"3. Horizontal vel (Vx):   {obs[2]:.2f}")
    print(f"4. Vertical vel (Vy):     {obs[3]:.2f}")
    print(f"5. Lander angle (rad):    {obs[4]:.2f}")
    print(f"6. Angular velocity:      {obs[5]:.2f}")
    print(f"7. Left leg contact:      {obs[6]:.0f}")
    print(f"8. Right leg contact:     {obs[7]:.0f}\n")

    # Run a short test with random actions
    # Actions: 0 = Do nothing, 1 = Fire left, 2 = Fire main, 3 = Fire right
    for step in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended at step {step}. Resetting environment...")
            obs, info = env.reset()

    env.close()