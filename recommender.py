import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from difflib import get_close_matches

def load_preprocessed_imdb_data():
    movies = pd.read_pickle('movies.pkl')
    with open('titles.pkl', 'rb') as f:
        movie_titles = pickle.load(f)
    return movies, movie_titles

def build_imdb_recommender(movies):
    movies['features'] = movies['genres_str'] + ' ' + movies['director_names']
    movies['features'] = movies['features'].str.lower()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    feature_vectors = vectorizer.fit_transform(movies['features'])
    return vectorizer, feature_vectors

def weighted_rating(x, m, C):
    v = x['numVotes']
    R = x['averageRating']
    score = (v / (v + m) * R) + (m / (v + m) * C)
    return score

def recommend_by_genres(selected_genres, movies, k=10):
    C = movies['averageRating'].mean()
    m = movies['numVotes'].quantile(0.70)
    def has_genres(genres):
        return all([genre in genres for genre in selected_genres])
    filtered_movies = movies[(movies['genres'].apply(has_genres)) & (movies['numVotes'] >= m)].copy()
    filtered_movies['score'] = filtered_movies.apply(weighted_rating, axis=1, m=m, C=C)
    filtered_movies = filtered_movies.sort_values('score', ascending=False)
    top_n = filtered_movies.head(k)
    return top_n
