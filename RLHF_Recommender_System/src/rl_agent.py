from stable_baselines3 import PPO

def train_agent(env, timesteps=10000):
    """
    Train PPO agent on the provided environment.
    """
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model
