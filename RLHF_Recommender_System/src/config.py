import os

# Dataset Configuration
DATASET_PATH = os.path.join("data", "raw", "ratings.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "preprocessed_data.csv")

# Environment Configuration
NUM_USERS = 1000     # Limit the number of users to speed up initial tests
NUM_MOVIES = 1000    # Limit number of movies for lightweight training
MAX_EPISODES = 10000  # Maximum environment episodes

# Reinforcement Learning Hyperparameters
RL_AGENT_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "verbose": 1,
}

# Training Configuration
NUM_TIMESTEPS = 50000  # Number of training steps
MODEL_SAVE_PATH = os.path.join("models", "best_rl_model.zip")
LOGGING_DIR = os.path.join("logs", "training_logs")

# Evaluation Configuration
NUM_EVAL_EPISODES = 10

# Neural Network Reward Model Hyperparameters
REWARD_MODEL_PARAMS = {
    "hidden_units": [64, 32],
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "loss": "mse",
}

# Logging and Debugging
LOGGING_LEVEL = "INFO"

# Performance Configuration
SEED_VALUE = 42
