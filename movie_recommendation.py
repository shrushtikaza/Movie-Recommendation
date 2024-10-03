import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def clean_title(title):
    """Clean the movie title by removing non-alphanumeric characters."""
    return re.sub("[^a-zA-Z0-9 ]", "", title)

# Load the movie dataset
try:
    df1 = pd.read_csv("movies.csv")
except FileNotFoundError:
    print("Error: movies.csv file not found.")
    exit()

# Clean the genres
df1['genres_list'] = df1['genres'].str.replace('\\|', ' ', regex=True)
df1['clean_title'] = df1['title'].apply(clean_title)

# Prepare the movies data
movies_data = df1[['movieId', 'clean_title', 'genres_list']]

# Vectorization for titles
vectorizer_title = TfidfVectorizer(ngram_range=(1, 2))
tfidf_title = vectorizer_title.fit_transform(movies_data['clean_title'])

def search_by_title(title):
    """Search for movies by title and return the top similar movies."""
    title = clean_title(title)
    query_vec = vectorizer_title.transform([title])
    similarity = cosine_similarity(query_vec, tfidf_title).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies_data.iloc[indices][::-1]  # Get the top 5 results
    return results

# Vectorization for genres
vectorizer_genres = TfidfVectorizer(ngram_range=(1, 2))
tfidf_genres = vectorizer_genres.fit_transform(movies_data['genres_list'])

def search_similar_genres(genres):
    """Search for movies with similar genres and return the top similar movies."""
    query_vec = vectorizer_genres.transform([genres])
    similarity = cosine_similarity(query_vec, tfidf_genres).flatten()
    indices = np.argpartition(similarity, -10)[-10:]
    results = movies_data.iloc[indices][::-1]  # Get the top 10 results
    return results

def recommendation_results(user_input):
    """Generate movie recommendations based on user input."""
    title_candidates = search_by_title(user_input)
    
    print("Are you looking for (please choose a number): ")
    for i in range(len(title_candidates)):
        print(i, ": ", title_candidates['clean_title'].iloc[i])
    
    title_index = int(input("Enter the number of the movie you are looking for: "))
    
    if title_index in range(len(title_candidates)):
        selected_movie_id = title_candidates.iloc[title_index]['movieId']
        selected_genres = title_candidates.iloc[title_index]['genres_list']
        
        # Get recommendations based on similar genres
        similar_genre_movies = search_similar_genres(selected_genres)
        
        print("\nWe have the following recommendations based on genres:")
        print(similar_genre_movies[['clean_title', 'genres_list']])
    else:
        print("Sorry! Please try again.")

# User input for movie title
user_input = input("Enter a movie title to get recommendations: ")
recommendation_results(user_input)
