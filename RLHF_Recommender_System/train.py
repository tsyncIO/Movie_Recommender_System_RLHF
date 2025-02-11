import pandas as pd
from src.environment import MovieRecommenderEnv
from src.rl_agent import train_agent
from src.data_loader import load_movielens_data

def main():
    # Load data
    data = load_movielens_data('data/raw/ratings.csv')

    # Create environment
    env = MovieRecommenderEnv(data)

    # Train agent
    model = train_agent(env)

    # Save model
    model.save("models/best_rl_model.zip")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
