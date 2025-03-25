import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Chemin vers les données (à ajuster selon votre configuration)
import kagglehub
path = kagglehub.dataset_download("sriharshabsprasad/movielens-dataset-100k-ratings")
movies_path = f"{path}/ml-latest-small/movies.csv"

# 1. Charger les données
print("Chargement des données...")
movies_df = pd.read_csv(movies_path)

# Afficher quelques infos sur les données
print(f"Nombre total de films: {len(movies_df)}")
print(movies_df.head())

# 2. Préparation des données pour TF-IDF
# Dans MovieLens, nous n'avons pas de descriptions textuelles, mais nous avons les genres
# Nous utiliserons les genres comme "contenu" pour notre système de recommandation
print("\nPréparation des données pour TF-IDF...")

# 3. Utilisation de TF-IDF pour représenter les genres des films
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'].str.replace('|', ' '))

print(f"Dimensions de la matrice TF-IDF: {tfidf_matrix.shape}")
print(f"Nombre de caractéristiques extraites: {len(tfidf.get_feature_names_out())}")
print("Termes extraits des genres:", tfidf.get_feature_names_out())

# 4. Calcul de la similarité entre les films
print("\nCalcul de la similarité entre les films...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Dimensions de la matrice de similarité: {cosine_sim.shape}")

# Création d'un DataFrame pour faciliter les recherches
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# 5. Fonction de recommandation de films similaires
def get_recommendations(title, cosine_sim=cosine_sim, df=movies_df, indices=indices):
    # Obtenir l'index du film
    try:
        idx = indices[title]
    except KeyError:
        print(f"Film '{title}' non trouvé dans la base de données.")
        return pd.DataFrame()
    
    # Obtenir les scores de similarité avec tous les autres films
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Trier les films par score de similarité
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtenir les 6 films les plus similaires (le premier est le film lui-même)
    sim_scores = sim_scores[1:6]
    
    # Obtenir les indices des films
    movie_indices = [i[0] for i in sim_scores]
    
    # Retourner les 5 films les plus similaires avec leurs scores
    result = df.iloc[movie_indices].copy()
    result['similarity_score'] = [i[1] for i in sim_scores]
    return result[['title', 'genres', 'similarity_score']]

# 6. Tester le système avec quelques films populaires
print("\nDémonstration du système de recommandation:\n")

test_movies = [
    "Toy Story (1995)",
    "Pulp Fiction (1994)",
    "Matrix, The (1999)",
    "Shawshank Redemption, The (1994)",
    "Star Wars: Episode IV - A New Hope (1977)"
]

for movie in test_movies:
    try:
        print(f"\nFilms similaires à '{movie}':")
        recommendations = get_recommendations(movie)
        if not recommendations.empty:
            print(recommendations)
    except Exception as e:
        print(f"Erreur avec le film '{movie}': {e}")
        # Vérifier si le film existe dans le dataset
        if movie not in indices:
            similar_titles = movies_df[movies_df['title'].str.contains(movie.split('(')[0].strip())]
            if not similar_titles.empty:
                print(f"Films avec un titre similaire trouvés: {similar_titles['title'].tolist()}")

# 7. Avantages et limitations de l'approche basée sur le contenu
print("\nAvantages et limitations de l'approche basée sur le contenu:")
print("\nAvantages:")
print("1. Pas de démarrage à froid pour les nouveaux éléments: peut recommander des items sans historique d'utilisation")
print("2. Indépendant des autres utilisateurs: peut faire des recommandations personnalisées sans données d'autres utilisateurs")
print("3. Bonne transparence: peut expliquer pourquoi un item a été recommandé")
print("4. Peut recommander des items de niche ou non populaires si leurs attributs correspondent aux préférences")

print("\nLimitations:")
print("1. Démarrage à froid pour les nouveaux utilisateurs: difficulté à faire des recommandations sans profil établi")
print("2. Sur-spécialisation/manque de diversité: tendance à recommander des items très similaires")
print("3. Limité par les attributs/métadonnées disponibles: qualité dépend de la richesse des descriptions")
print("4. Ne capture pas les préférences qui ne sont pas exprimées dans les attributs des items")
print("5. Ne profite pas de la sagesse collective comme le font les approches collaboratives") 