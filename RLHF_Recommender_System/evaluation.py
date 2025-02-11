from stable_baselines3 import PPO
from src.environment import MovieRecommenderEnv
from src.data_loader import load_movielens_data

def evaluate_agent(model_path, env, num_episodes=10):
    """
    Evaluate trained RL agent.
    """
    model = PPO.load(model_path)
    total_rewards = 0

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_rewards += reward

    avg_reward = total_rewards / num_episodes
    print(f"Average Reward: {avg_reward}")

if __name__ == "__main__":
    data = load_movielens_data('data/raw/ratings.csv')
    env = MovieRecommenderEnv(data)
    evaluate_agent("models/best_rl_model.zip", env)
