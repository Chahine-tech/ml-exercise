import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Path to data (adjust according to your configuration)
import kagglehub
path = kagglehub.dataset_download("sriharshabsprasad/movielens-dataset-100k-ratings")
movies_path = f"{path}/ml-latest-small/movies.csv"

# 1. Load data
print("Loading data...")
movies_df = pd.read_csv(movies_path)

# Display some information about the data
print(f"Total number of movies: {len(movies_df)}")
print("\nSample of available movies:")
print(movies_df.head())

# 2. Prepare data for TF-IDF
# In MovieLens, we don't have text descriptions, but we have genres
# We will use genres as "content" for our recommendation system
print("\nPreparing data for TF-IDF...")

# 3. Use TF-IDF to represent movie genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'].str.replace('|', ' '))

print(f"TF-IDF matrix dimensions: {tfidf_matrix.shape}")
print(f"Number of extracted features: {len(tfidf.get_feature_names_out())}")
print("Terms extracted from genres:", tfidf.get_feature_names_out())

# 4. Calculate similarity between movies
print("\nCalculating similarity between movies...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Similarity matrix dimensions: {cosine_sim.shape}")

# Create a DataFrame to facilitate searches
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# 5. Function to recommend similar movies
def get_recommendations(title, cosine_sim=cosine_sim, df=movies_df, indices=indices):
    # Get the movie index
    try:
        idx = indices[title]
    except KeyError:
        print(f"Movie '{title}' not found in the database.")
        return pd.DataFrame()
    
    # Get similarity scores with all other movies
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort movies by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the 6 most similar movies (the first one is the movie itself)
    sim_scores = sim_scores[1:6]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the 5 most similar movies with their scores
    result = df.iloc[movie_indices].copy()
    result['similarity_score'] = [i[1] for i in sim_scores]
    return result[['title', 'genres', 'similarity_score']]

# 6. Test the system with random movies from the dataset
print("\nTesting the recommendation system with random movies from the dataset:\n")

# Select 5 random movies from the dataset
random_movies = movies_df.sample(n=5, random_state=42)

for _, movie in random_movies.iterrows():
    title = movie['title']
    print(f"\nMovies similar to '{title}':")
    recommendations = get_recommendations(title)
    if not recommendations.empty:
        print(recommendations)
    print("-" * 80)

# 7. Advantages and limitations of content-based approach
print("\nAdvantages and limitations of content-based approach:")
print("\nAdvantages:")
print("1. No cold start for new items: can recommend items without usage history")
print("2. Independent of other users: can make personalized recommendations without other users' data")
print("3. Good transparency: can explain why an item was recommended")
print("4. Can recommend niche or unpopular items if their attributes match preferences")

print("\nLimitations:")
print("1. Cold start for new users: difficulty in making recommendations without established profile")
print("2. Over-specialization/lack of diversity: tendency to recommend very similar items")
print("3. Limited by available attributes/metadata: quality depends on richness of descriptions")
print("4. Does not capture preferences that are not expressed in item attributes")
print("5. Does not benefit from collective wisdom like collaborative approaches") 