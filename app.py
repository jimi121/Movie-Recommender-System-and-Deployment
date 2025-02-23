import random  # Import the random module for generating random numbers
import requests  # Import the requests module for making HTTP requests
import pickle  # Import the pickle module for serializing and deserializing Python objects
import pandas as pd  # Import the pandas module and alias it as pd for data manipulation
from flask import Flask, jsonify, render_template, request, url_for  # Import specific functions from Flask for web development
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer for text feature extraction
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity for calculating cosine similarity between vectors
from sklearn.neighbors import NearestNeighbors  # Import NearestNeighbors for implementing the k-nearest neighbors algorithm
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler for scaling data

app = Flask(__name__)  # Create a Flask application instance
TMDB_API_KEY = '306c333178bf1802b38c1f8863f606fc'  # Define the TMDB API key


# Fetch movies from TMDb
def fetch_movies_from_tmdb():
    movies = []  # Initialize an empty list to store movie data
    for page in range(1, 6):  # Loop through the first 5 pages of popular movies
        url = f'https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page={page}'  # Define the URL for fetching popular movies
        response = requests.get(url)  # Send an HTTP GET request to the URL
        data = response.json()  # Parse the response as JSON
        movies.extend(data['results'])  # Extend the movies list with the results from the current page
    return pd.DataFrame(movies)  # Convert the movies list to a pandas DataFrame and return it


def fetch_movie_details(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US'  # Define the URL for fetching movie details
    response = requests.get(url)  # Send an HTTP GET request to the URL
    return response.json()  # Parse the response as JSON and return it


def fetch_movie_cast(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}&language=en-US'  # Define the URL for fetching movie cast
    response = requests.get(url)  # Send an HTTP GET request to the URL
    return response.json()['cast']  # Parse the response as JSON and return the cast


# Load movie data
movies_df = fetch_movies_from_tmdb()  # Fetch movies from TMDb and store them in a DataFrame
movies_df['details'] = movies_df['id'].apply(fetch_movie_details)  # Apply fetch_movie_details to each movie and store the result in a new column
movies_df['genres'] = movies_df['details'].apply(lambda x: ', '.join([genre['name'] for genre in x['genres']]))  # Extract genres from details and join them into a single string
movies_df['overview'] = movies_df['overview'].fillna('')  # Fill missing overviews with an empty string
movies_df['content'] = movies_df['genres'] + ' ' + movies_df['overview']  # Combine genres and overview into a single content column

# Build content-based filtering model
tfidf = TfidfVectorizer(stop_words='english')  # Initialize a TF-IDF Vectorizer with English stop words
tfidf_matrix = tfidf.fit_transform(movies_df['content'])  # Fit and transform the content column to a TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)  # Compute cosine similarity between all movie vectors


# Function to get content-based recommendations
def get_content_based_recommendations(movie_id, cosine_sim=cosine_sim):
    if movie_id not in movies_df['id'].values:  # Check if the movie ID is in the DataFrame
        return []  # Return an empty list if not found
    idx = movies_df.index[movies_df['id'] == movie_id].tolist()[0]  # Get the index of the movie
    sim_scores = list(enumerate(cosine_sim[idx]))  # Get similarity scores for the movie
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort similarity scores in descending order
    sim_scores = sim_scores[1:11]  # Get the top 10 similar movies (excluding the movie itself)
    movie_indices = [i[0] for i in sim_scores]  # Extract movie indices from the similarity scores
    return movies_df['id'].iloc[movie_indices].tolist()  # Return the IDs of the recommended movies


# Fetch ratings for collaborative filtering
def fetch_ratings():
    ratings = []  # Initialize an empty list to store rating data
    for page in range(1, 6):  # Loop through the first 5 pages of top-rated movies
        url = f'https://api.themoviedb.org/3/movie/top_rated?api_key={TMDB_API_KEY}&language=en-US&page={page}'  # Define the URL for fetching top-rated movies
        response = requests.get(url)  # Send an HTTP GET request to the URL
        data = response.json()  # Parse the response as JSON
        ratings.extend(data['results'])  # Extend the ratings list with the results from the current page
    return pd.DataFrame(ratings)  # Convert the ratings list to a pandas DataFrame and return it


# Prepare data for collaborative filtering
ratings_df = fetch_ratings()  # Fetch ratings from TMDb and store them in a DataFrame
ratings_df['user_id'] = ratings_df.index  # Assign user IDs based on the DataFrame index
ratings_matrix = pd.pivot_table(ratings_df, values='vote_average', index='user_id', columns='id').fillna(0)  # Pivot the DataFrame to create a user-item matrix and fill missing values with 0

# Normalize ratings
scaler = MinMaxScaler()  # Initialize a MinMaxScaler
ratings_matrix = scaler.fit_transform(ratings_matrix)  # Scale the ratings matrix to the range [0, 1]

# Build k-NN model for collaborative filtering
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)  # Initialize a k-NN model with cosine similarity and brute-force search
knn.fit(ratings_matrix)  # Fit the k-NN model to the ratings matrix


# Function to get collaborative recommendations
def get_collaborative_recommendations(movie_id):
    if movie_id not in ratings_df['id'].values:
        return []

    try:
        movie_idx = list(ratings_df['id']).index(movie_id)
        distances, indices = knn.kneighbors(ratings_matrix[:, movie_idx].reshape(1, -1), n_neighbors=11)
        movie_indices = [ratings_df['id'].iloc[i] for i in indices.flatten()][1:]
        return movie_indices
    except ValueError as e:
        print(f"Error: {e}")
        return []


# Combine both approaches for recommendations
def get_combined_recommendations(movie_id):
    content_based_recs = get_content_based_recommendations(movie_id)  # Get content-based recommendations
    collaborative_recs = get_collaborative_recommendations(movie_id)  # Get collaborative recommendations
    combined_recs = list(set(content_based_recs + collaborative_recs))  # Combine and deduplicate the recommendations
    if len(combined_recs) == 0:  # If no recommendations found in DataFrame, fetch from TMDb
        url = f'https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={TMDB_API_KEY}&language=en-US'  # Define the URL for fetching movie recommendations
        response = requests.get(url)  # Send an HTTP GET request to the URL
        data = response.json()  # Parse the response as JSON
        combined_recs = [movie['id'] for movie in data['results']]  # Extract movie IDs from the results
    return combined_recs[:20]  # Return the top 10 recommendations


# Save the model
with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump({
        'tfidf': tfidf,
        'cosine_sim': cosine_sim,
        'knn': knn,
        'movies_df': movies_df
    }, f)  # Serialize and save the recommendation model components to a file

# Load the recommendation model
with open('recommendation_model.pkl', 'rb') as f:
    recommendation_model = pickle.load(f)  # Load the recommendation model components from the file

tfidf = recommendation_model['tfidf']  # Assign the loaded TF-IDF vectorizer to a variable
cosine_sim = recommendation_model['cosine_sim']  # Assign the loaded cosine similarity matrix to a variable
knn = recommendation_model['knn']  # Assign the loaded k-NN model to a variable
movies_df = recommendation_model['movies_df']  # Assign the loaded movies DataFrame to a variable


def fetch_trailer(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}&language=en-US'  # Define the URL for fetching movie trailers
    response = requests.get(url)  # Send an HTTP GET request to the URL
    data = response.json()  # Parse the response as JSON
    trailers = [video for video in data.get('results', []) if video['type'] == 'Trailer' and video['site'] == 'YouTube']  # Filter trailers from the results
    for trailer in trailers:
        if trailer:
            return f"https://www.youtube.com/embed/{trailer['key']}"  # Return the YouTube embed URL for the trailer
    return None  # Return None if no trailers found


def get_full_language_name(language_code):
    languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'ja': 'Japanese',
        'zh': 'Chinese',
        # Add other languages as needed
    }  # Define a dictionary mapping language codes to full names
    return languages.get(language_code, language_code)  # Return the full language name or the code if not found


@app.route('/')
def index():
    for page in range(1, 11):
        popular_movies_url = f'https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page={page}'  # Define the URL for fetching popular movies
        response = requests.get(popular_movies_url)  # Send an HTTP GET request to the URL
        data = response.json()  # Parse the response as JSON
        movies = [movie for movie in data.get('results', []) if movie['poster_path']]  # Filter movies with poster paths
        for movie in movies:
            trailer_url = fetch_trailer(movie['id'])  # Fetch the trailer URL for the movie
            if trailer_url:
                return render_template('index.html', trailer_url=trailer_url, trailer_title=movie['title'])  # Render the index template with trailer data
    return render_template('index.html', trailer_url=None, trailer_title='')  # Render the index template without trailer data


@app.route('/search')
def search():
    query = request.args.get('query', '')  # Get the search query from the request arguments
    search_url = f'https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&language=en-US&query={query}&page=1&include_adult=false'  # Define the URL for searching movies
    response = requests.get(search_url)  # Send an HTTP GET request to the URL
    data = response.json()  # Parse the response as JSON
    results = data.get('results', [])  # Get the search results
    valid_results = [movie for movie in results if movie['poster_path']]  # Filter movies with poster paths

    trailer_url = None
    trailer_title = ''
    for movie in valid_results:
        trailer_url = fetch_trailer(movie['id'])  # Fetch the trailer URL for the movie
        trailer_title = movie['title']
        if trailer_url:
            break

    return jsonify({
        'results': valid_results,
        'trailer_url': trailer_url,
        'trailer_title': trailer_title
    })  # Return the search results and trailer data as JSON


@app.route('/default_movies')
def default_movies():
    trending_movies = []
    for page in range(1, 11):
        popular_movies_url = f'https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page={page}'  # Define the URL for fetching popular movies
        response = requests.get(popular_movies_url)  # Send an HTTP GET request to the URL
        data = response.json()  # Parse the response as JSON
        trending_movies.extend([movie for movie in data.get('results', []) if movie['poster_path']])  # Filter and extend the trending movies list
        if len(trending_movies) >= 200:
            break

    return jsonify({
        'trending_movies': trending_movies[:200]
    })  # Return the top 200 trending movies as JSON


@app.route('/movie_reviews/<int:movie_id>')
def fetch_movie_reviews(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}&language=en-US'
    response = requests.get(url)
    data = response.json()

    reviews = []
    for review in data.get('results', []):
        rating = review.get('author_details', {}).get('rating', None)  # Fetch rating if available
        reviews.append({
            'author': review.get('author', 'Anonymous'),
            'content': review.get('content', 'No review content available.'),
            'rating': rating
        })

    return jsonify(reviews)

@app.route('/top_ten_movies')
def top_ten_movies():
    top_ten_movies = []
    for page in range(1, 2):
        top_ten_movies_url = f'https://api.themoviedb.org/3/movie/top_rated?api_key={TMDB_API_KEY}&language=en-US&page={page}'  # Define the URL for fetching top-rated movies
        response = requests.get(top_ten_movies_url)  # Send an HTTP GET request to the URL
        data = response.json()  # Parse the response as JSON
        top_ten_movies.extend([movie for movie in data.get('results', []) if movie['poster_path']])  # Filter and extend the top 10 movies list
        if len(top_ten_movies) >= 10:
            break

    return jsonify({
        'top_ten_movies': top_ten_movies[:20]
    })  # Return the top 10 movies as JSON


@app.route('/recommend', methods=['POST'])
def recommend():
    movie_id = request.json.get('movie_id')  # Get the movie ID from the request JSON
    recommendations = get_combined_recommendations(movie_id)  # Get combined recommendations for the movie

    recommend_movies = []
    for idx in recommendations:
        movie_url = f'https://api.themoviedb.org/3/movie/{idx}?api_key={TMDB_API_KEY}&language=en-US'  # Define the URL for fetching movie details
        response = requests.get(movie_url)  # Send an HTTP GET request to the URL
        movie_data = response.json()  # Parse the response as JSON
        if 'poster_path' in movie_data and movie_data['poster_path']:
            recommend_movies.append(movie_data)  # Append movie data to the recommended movies list

    main_movie_url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US'  # Define the URL for fetching the main movie details
    response = requests.get(main_movie_url)  # Send an HTTP GET request to the URL
    main_movie = response.json()  # Parse the response as JSON

    return jsonify({
        'main_movie': main_movie,
        'similar_movies': recommend_movies
    })  # Return the main movie and similar movies as JSON


@app.route('/movie_details/<int:movie_id>')
def movie_details(movie_id):
    movie_url = f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US'
    response = requests.get(movie_url)
    movie_data = response.json()

    cast_url = f'https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}&language=en-US'
    response = requests.get(cast_url)
    cast_data = response.json()
    top_cast = [cast['name'] for cast in cast_data['cast'][:5]]
    top_cast_photos = [{'name': cast['name'], 'character': cast['character'], 'profile_path': cast['profile_path'], 'id': cast['id']} for cast in cast_data['cast'][:5] if cast['profile_path']]

    trailer_url = fetch_trailer(movie_id)
    if not trailer_url:
        trailer_url = url_for('static', filename='img/img2.png')

    reviews_url = f'https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={TMDB_API_KEY}&language=en-US'
    response = requests.get(reviews_url)
    reviews_data = response.json()
    reviews = reviews_data.get('results', [])

    trending_movies = []
    for page in range(1, 11):
        popular_movies_url = f'https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page={page}'  # Define the URL for fetching popular movies
        response = requests.get(popular_movies_url)  # Send an HTTP GET request to the URL
        data = response.json()  # Parse the response as JSON
        trending_movies.extend([movie for movie in data.get('results', []) if
                                movie['poster_path']])  # Filter and extend the trending movies list
        if len(trending_movies) >= 200:
            break

    genre_names = [genre['name'] for genre in movie_data['genres']]  # Convert genre IDs to names
    movie_data['genres'] = ', '.join(genre_names)  # Join the genre names into a single string

    movie_data['original_language'] = get_full_language_name(
        movie_data['original_language'])  # Convert language code to full name

    recommend_movies = get_combined_recommendations(movie_id)  # Get combined recommendations for the movie
    recommend_movie_data = []
    for idx in recommend_movies:
        movie_url = f'https://api.themoviedb.org/3/movie/{idx}?api_key={TMDB_API_KEY}&language=en-US'  # Define the URL for fetching movie details
        response = requests.get(movie_url)  # Send an HTTP GET request to the URL
        rec_movie_data = response.json()  # Parse the response as JSON
        if 'poster_path' in rec_movie_data and rec_movie_data['poster_path']:
            recommend_movie_data.append(rec_movie_data)  # Append movie data to the recommended movies list

    return render_template('movie_details.html', movie=movie_data, top_cast=top_cast, top_cast_photos=top_cast_photos, trailer_url=trailer_url, reviews=reviews, trending_movies=trending_movies[:200], recommend_movies=recommend_movie_data)  # Render the movie details template with the movie data

@app.route('/cast_details/<int:cast_id>')
def fetch_cast_details(cast_id):
    url = f'https://api.themoviedb.org/3/person/{cast_id}?api_key={TMDB_API_KEY}&language=en-US'
    response = requests.get(url)
    data = response.json()
    return jsonify(data)

@app.route('/people_also_watch')
def people_also_watch():
    all_movies = []
    for page in range(1, 21):
        popular_movies_url = f'https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page={page}'  # Define the URL for fetching popular movies
        response = requests.get(popular_movies_url)  # Send an HTTP GET request to the URL
        data = response.json()  # Parse the response as JSON
        all_movies.extend([movie for movie in data.get('results', []) if movie['poster_path']])  # Filter and extend the all movies list
        if len(all_movies) >= 600:
            break

    all_movies = {movie['id']: movie for movie in all_movies}.values()  # Remove duplicates and keep the first 500 unique movies
    return jsonify({
        'people_also_watch': list(all_movies)[:500]
    })  # Return the top 500 movies as JSON

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask application in debug mode
