import kagglehub
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# 1. Download dataset (already done)
path = kagglehub.dataset_download("sriharshabsprasad/movielens-dataset-100k-ratings")
print("Path to dataset files:", path)

# Define paths to the dataset files
ratings_path = f"{path}/ml-latest-small/ratings.csv"
movies_path = f"{path}/ml-latest-small/movies.csv"

# 2. Read the data
print("Loading data...")
ratings_df = pd.read_csv(ratings_path)
movies_df = pd.read_csv(movies_path)

# Show dataset dimensions
print(f"Loaded {len(ratings_df)} ratings across {len(movies_df)} movies")
print(f"Number of unique users: {ratings_df['userId'].nunique()}")
print(f"Number of unique movies: {ratings_df['movieId'].nunique()} (rated movies)")
print(f"Total movies in movies.csv: {len(movies_df)}")

# Display the first few rows of each dataframe
print("\nSample of ratings data:")
print(ratings_df.head())

print("\nSample of movies data:")
print(movies_df.head())

# 3. Create the utility matrix (users in rows, movies in columns)
print("\nCreating utility matrix...")
utility_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print(f"Utility matrix shape: {utility_matrix.shape} (users × movies)")

# Display a small section of the utility matrix
print("\nSample of utility matrix (5×5):")
print(utility_matrix.iloc[:5, :5])

# 4. Calculate similarities
print("\nCalculating similarities...\n")

# 4.1 Select two users for comparison
user1_id = utility_matrix.index[0]  # First user
user2_id = utility_matrix.index[1]  # Second user

user1_ratings = utility_matrix.loc[user1_id].values.reshape(1, -1)
user2_ratings = utility_matrix.loc[user2_id].values.reshape(1, -1)

# 4.2 Select two movies for comparison
movie1_id = utility_matrix.columns[0]  # First movie
movie2_id = utility_matrix.columns[1]  # Second movie

movie1_ratings = utility_matrix[movie1_id].values.reshape(-1, 1)
movie2_ratings = utility_matrix[movie2_id].values.reshape(-1, 1)

# 4.3 Calculate Cosine Similarity
# Between users
user_cosine_sim = cosine_similarity(user1_ratings, user2_ratings)[0][0]
print(f"Cosine similarity between User {user1_id} and User {user2_id}: {user_cosine_sim:.4f}")

# Between movies
movie_cosine_sim = cosine_similarity(movie1_ratings, movie2_ratings)[0][0]
print(f"Cosine similarity between Movie {movie1_id} and Movie {movie2_id}: {movie_cosine_sim:.4f}")

# 4.4 Calculate Pearson Correlation
# Filter out zero ratings (where either user hasn't rated the movie)
# For users
common_movies_mask = (user1_ratings[0] != 0) & (user2_ratings[0] != 0)
if sum(common_movies_mask) > 1:  # Need at least 2 points for correlation
    user_pearson_corr, _ = pearsonr(
        user1_ratings[0][common_movies_mask], 
        user2_ratings[0][common_movies_mask]
    )
    print(f"Pearson correlation between User {user1_id} and User {user2_id}: {user_pearson_corr:.4f}")
    print(f"Number of common rated movies: {sum(common_movies_mask)}")
else:
    print(f"Insufficient common movies between User {user1_id} and User {user2_id} for Pearson correlation")

# For movies
common_users_mask = (movie1_ratings != 0).flatten() & (movie2_ratings != 0).flatten()
if sum(common_users_mask) > 1:  # Need at least 2 points for correlation
    movie_pearson_corr, _ = pearsonr(
        movie1_ratings.flatten()[common_users_mask],
        movie2_ratings.flatten()[common_users_mask]
    )
    print(f"Pearson correlation between Movie {movie1_id} and Movie {movie2_id}: {movie_pearson_corr:.4f}")
    print(f"Number of common users who rated both movies: {sum(common_users_mask)}")
else:
    print(f"Insufficient common users between Movie {movie1_id} and Movie {movie2_id} for Pearson correlation")

# 5. Interpretation
print("\nInterpretation of results:")
print("Cosine similarity measures the cosine of the angle between two vectors,")
print("ranging from -1 (opposite) to 1 (same direction), with 0 indicating orthogonality (no relationship).")
print("Pearson correlation measures linear relationship between variables,")
print("ranging from -1 (perfect negative correlation) to 1 (perfect positive correlation).")

print("\nFor users:")
if 'user_cosine_sim' in locals():
    if user_cosine_sim > 0.7:
        print(f"Cosine similarity of {user_cosine_sim:.4f} indicates users have very similar preferences")
    elif user_cosine_sim > 0.3:
        print(f"Cosine similarity of {user_cosine_sim:.4f} indicates users have somewhat similar preferences")
    else:
        print(f"Cosine similarity of {user_cosine_sim:.4f} indicates users have different preferences")

if 'user_pearson_corr' in locals():
    if abs(user_pearson_corr) > 0.7:
        print(f"Pearson correlation of {user_pearson_corr:.4f} indicates a strong {'positive' if user_pearson_corr > 0 else 'negative'} linear relationship")
    elif abs(user_pearson_corr) > 0.3:
        print(f"Pearson correlation of {user_pearson_corr:.4f} indicates a moderate {'positive' if user_pearson_corr > 0 else 'negative'} linear relationship")
    else:
        print(f"Pearson correlation of {user_pearson_corr:.4f} indicates a weak linear relationship")

print("\nFor movies:")
if 'movie_cosine_sim' in locals():
    if movie_cosine_sim > 0.7:
        print(f"Cosine similarity of {movie_cosine_sim:.4f} indicates movies are rated similarly by users")
    elif movie_cosine_sim > 0.3:
        print(f"Cosine similarity of {movie_cosine_sim:.4f} indicates movies are rated somewhat similarly")
    else:
        print(f"Cosine similarity of {movie_cosine_sim:.4f} indicates movies are rated differently")

if 'movie_pearson_corr' in locals():
    if abs(movie_pearson_corr) > 0.7:
        print(f"Pearson correlation of {movie_pearson_corr:.4f} indicates a strong {'positive' if movie_pearson_corr > 0 else 'negative'} linear relationship")
    elif abs(movie_pearson_corr) > 0.3:
        print(f"Pearson correlation of {movie_pearson_corr:.4f} indicates a moderate {'positive' if movie_pearson_corr > 0 else 'negative'} linear relationship")
    else:
        print(f"Pearson correlation of {movie_pearson_corr:.4f} indicates a weak linear relationship")