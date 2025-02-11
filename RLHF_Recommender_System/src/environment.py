import gym
from gym import spaces
import numpy as np

class MovieRecommenderEnv(gym.Env):
    """
    Custom Gym environment simulating a movie recommendation scenario.
    """
    def __init__(self, data, num_users=100, num_movies=100):
        super(MovieRecommenderEnv, self).__init__()

        self.data = data
        self.num_users = min(num_users, data['userId'].nunique())
        self.num_movies = min(num_movies, data['movieId'].nunique())
        self.action_space = spaces.Discrete(self.num_movies)
        self.observation_space = spaces.Discrete(self.num_users)
        self.state = None

    def reset(self):
        self.state = np.random.choice(self.num_users)
        return self.state

    def step(self, action):
        reward = self._calculate_reward(self.state, action)
        done = True
        return self.state, reward, done, {}

    def _calculate_reward(self, user, movie):
        """
        Reward calculation based on user feedback data.
        """
        user_data = self.data[self.data.userId == user]
        if movie in user_data.movieId.values:
            return user_data[user_data.movieId == movie].rating.values[0]
        return 0.0
