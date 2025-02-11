import os
import pandas as pd

def load_movielens_data(filepath='data/raw/ratings.csv'):
    """
    Loads and validates the MovieLens dataset.
    
    Args:
        filepath (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Preprocessed DataFrame containing userId, movieId, rating.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at path: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Basic validation
    required_cols = {'userId', 'movieId', 'rating'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    # Drop missing values and reset index
    df = df[['userId', 'movieId', 'rating']].dropna().reset_index(drop=True)
    print(f"Loaded dataset with {len(df)} records.")
    return df
