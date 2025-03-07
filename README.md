# 🎥 MovixCloud: A Hybrid Movie Recommendation System

![](https://github.com/jimi121/Movie-Recommender-System-and-Deployment/blob/main/static/img/image.png)

An end-to-end **hybrid movie recommendation system** combining content-based and collaborative filtering techniques to deliver personalized movie suggestions. This project demonstrates expertise in data acquisition, preprocessing, modeling, evaluation, and deployment.

## Table of Contents

1. [Overview & Motivation](#overview--motivation)
2. [Features](#features)
3. [Data Collection & Preprocessing](#data-collection--preprocessing)
4. [Methodology](#methodology)
5. [Model Evaluation](#model-evaluation)
6. [Deployment & Model Serialization](#deployment--model-serialization)
7. [Project Structure](#project-structure)
8. [Prerequisites](#prerequisites)
9. [Installation & Deployment](#installation--deployment)
10. [Usage](#usage)
11. [Demo](#demo)
12. [API Endpoints](#api-endpoints)
13. [Future Work](#future-work)
14. [Conclusion](#conclusion)
15. [Acknowledgments](#acknowledgments)
16. [Contact](#contact)

---

## Overview & Motivation

The goal of this project is to build a robust recommendation engine that helps users discover movies they’ll love by leveraging two complementary approaches:

- **Content-Based Filtering:** Uses textual features (genres and movie overviews) transformed with TF-IDF to capture thematic similarities between movies.
- **Collaborative Filtering:** Leverages user ratings to identify patterns in movie preferences using a k-Nearest Neighbors (k-NN) approach.

By merging these methods into a hybrid system, the solution addresses challenges like the cold-start problem and data sparsity while enhancing overall recommendation quality.

---

## Features

- **Hybrid Recommendations:** Combines content-based and collaborative filtering for better personalization.
- **Real-Time Suggestions:** Provides instant recommendations based on user input.
- **Fallback Mechanism:** Handles cases where data is insufficient by fetching recommendations directly from TMDb.
- **Real-Time Data Integration**:  
  Uses live data from the TMDB API to ensure the recommendations are based on current movie information.

- **Simplified Web Interface**:  
  A basic Flask-based interface lets users search for movies and view details. (Note: Web development is not my primary strength, so the UI is kept minimal to emphasize the ML work.)

- **Embedded Trailer & Cast Details**:  
  Shows YouTube trailers and actor information, adding context to the recommendations.

- **Interactive Search & Reviews**:  
  Users can search for movies and submit reviews, providing additional feedback channels.

- **Deployment Ready**:  
  The project is deployed on a cloud platform (link provided) demonstrating my ability to take a data science project from concept to live application.

---

## Data Collection & Preprocessing

### Data Acquisition

- **Data Source:**  
  Movie data (popular movies, detailed metadata, cast information, and ratings) was obtained via the [TMDb API](https://www.themoviedb.org/documentation/api).
- **APIs Used:**  
  REST endpoints were used to fetch data on popular movies, top-rated movies, and movie details.

### Preprocessing Steps

- **Data Cleaning:**  
  Missing data was handled by filling null fields (e.g., replacing missing overviews with empty strings) and filtering out movies lacking critical assets like poster images.
- **Feature Engineering:**  
  Genres and overviews were merged into a single “content” field, enabling natural language processing techniques.
- **User-Item Matrix:**  
  A pivot table was created from ratings data, with missing ratings filled using zeros and normalized using MinMaxScaler.

---

## Methodology

### Content-Based Filtering

- **TF-IDF Vectorization:**  
  Transforms combined text (genres and overviews) into numerical vectors, highlighting unique terms that capture the movie’s theme.
- **Cosine Similarity:**  
  Measures similarity between TF-IDF vectors to recommend movies with similar content.

### Collaborative Filtering

- **K-NN Approach:**  
  Identifies movies with similar rating patterns across users using cosine similarity as the distance metric.
- **Handling Sparsity:**  
  Missing values in the user-item matrix are addressed by normalization, ensuring effective k-NN performance.

### Hybrid Recommendation

- **Merging Recommendations:**  
  Combines outputs from both models, removing duplicates to create a comprehensive set of personalized suggestions.

---

## **Model Evaluation** 
Since this is a prototype recommendation system, evaluation is currently **qualitative** rather than quantitative. Here’s how the system is validated:

1. **Manual Inspection**:  
   Recommendations are manually checked for relevance (e.g., if a user likes *The Avengers*, the system should suggest similar action/superhero movies).  
   Example: For `movie_id=533535` (a superhero movie), recommendations are inspected to ensure they align with genres like "Action" or "Sci-Fi."

2. **Similarity Score Analysis**:  
   The cosine similarity matrix (`cosine_sim`) is analyzed to verify that movies with overlapping genres/overviews have higher similarity scores (e.g., *Inception* and *Interstellar* should have high similarity).

3. **Hybrid Approach Validation**:  
   The combined recommendations (content-based + collaborative) are checked for diversity and relevance. For instance, collaborative filtering might surface niche movies, while content-based filtering ensures genre alignment.

---

## Deployment & Model Serialization

- **Serialization with Pickle:**  
  Key model components are serialized using Python’s pickle module for fast loading during runtime.
- **Flask Integration:**  
  The recommendation engine is exposed via a Flask API, making it accessible as a microservice.

---

## Project Structure

```
├── app.py                # Flask application entry point
├── recommendation_model.pkl  # Serialized model components
├── static/               # Static assets (CSS, JS, images, fonts)
├── templates/            # HTML templates for the UI
├── README.md             # Project documentation
└── requirements.txt      # Dependencies and library versions
```

---
## Prerequisites

To run the project locally, you need:

- **Python (>= 3.9)**
- **pip** (Python package manager)
- **TMDB API Key**: Register at [TMDB](https://www.themoviedb.org/) to obtain an API key.

Install dependencies with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` isn’t available, manually install:

```bash
pip install flask requests pandas scikit-learn pickle-mixin
```

---

## Installation & Deployment

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Configure Environment Variables:** 

   Create a `.env` file in the root directory and add your TMDB API key:

   ```ini
   TMDB_API_KEY=your_api_key_here
   ```

3. **Run the Application:**

   Start the Flask server:

   ```bash
   python app.py
   ```

4. **Access the Application:**

   Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000).

5. **Deployed Version:**

   Check out the live version here: [try the Project live here](https://movixcloud.onrender.com)

---

## Usage

### How It Works

- **Content-Based Filtering**:  
  Analyzes movie metadata to suggest similar movies.  
  _Example: Enjoying *Inception* may lead to recommendations for *Interstellar*._

- **Collaborative Filtering**:  
  Uses rating data to identify similar user preferences and suggest movies accordingly.  
  _Example: Users who liked *Avengers: Endgame* might also like *Iron Man*._

- **Hybrid Approach**:  
  Combines both methods to offer a balanced set of recommendations.

### Sample Workflow

1. **Search**:  
   Use the search bar to find a movie by title, genre, or keyword.
2. **View Details**:  
   Click on a movie to see its cast, embedded trailer, and additional recommendations.
3. **Explore**:  
   Browse trending or top-rated movies directly from the homepage.

---
## Demo ✨

### Live Application
Experience the project live by visiting the deployed version:  
[Check the Project ](https://movixcloud.onrender.com)

### Video Walkthrough
Below is a demonstration of project's key features, including hybrid recommendations, real-time data fetching, and interactive user interface:

![Project Demo](path/to/demo.gif)  
*Figure: A walkthrough of project's main functionalities.*

---
## API Endpoints

| **Endpoint**                     | **Method** | **Description**                                                                |
|----------------------------------|------------|--------------------------------------------------------------------------------|
| `/`                              | GET        | Displays popular movies and a featured trailer.                               |
| `/search`                        | GET        | Searches for movies by a query string and returns JSON suggestions.            |
| `/default_movies`                | GET        | Returns a list of 200 trending movies.                                         |
| `/top_twenty_movies`             | GET        | Lists the top 20 rated movies.                                                 |
| `/recommend`                     | POST       | Provides hybrid recommendations based on a given movie ID.                     |
| `/movie_details/`                | GET        | Retrieves detailed information about a specific movie.                         |
| `/people_also_watch`             | GET        | Suggests additional movies often watched together.                             |
| `/cast_details/`                 | GET        | Fetches detailed bios for an actor.                                            |
| `/movie_reviews/`                | GET        | Retrieves user reviews from TMDB.                                              |
| `/submit_review`                 | POST       | Accepts user review submissions via the frontend.                              |

---

## Future Work

- **Enhanced Feature Engineering:**  
  Experiment with advanced NLP models like Word2Vec, BERT, or transformers for richer semantic representations.
- **Advanced Collaborative Techniques:**  
  Explore matrix factorization (e.g., SVD) or deep learning models to capture latent factors in user preferences.
- **Scalability:**  
  Implement distributed computing frameworks (e.g., Apache Spark) and approximate nearest neighbor libraries (e.g., FAISS) for large-scale deployments.
- **Continuous Learning:**  
  Integrate user interaction data for dynamic model updates and improved personalization.
 
To rigorously evaluate performance, I plan to implement:  
- **Offline Metrics**: Precision@k, Recall@k, NDCG (to measure ranking quality).  
- **A/B Testing**: Compare user engagement with recommendations from this system vs. a baseline (e.g., popularity-based recommendations).

---

## Conclusion

This project showcases my ability to build a scalable, hybrid recommendation engine using advanced data science techniques. It highlights skills in data acquisition, natural language processing, machine learning, and model deployment.

---

## Acknowledgments

- **TMDB**: Thanks to [The Movie Database](https://www.themoviedb.org/) for providing movie data.
- **Scikit-Learn & Flask**: For the robust tools that powered this project.
- **Open-Source Community**: For constant inspiration and support in my data science journey.

---

### Contact

Feel free to reach out or contribute if you have any suggestions or improvements!

**Email:** olajimiadeleke4@gmail.com

**LinkedIn:** [LinkedIn Profile](https://www.linkedin.com/public-profile/settings?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_self_edit_contact-info%3BTyeFCIhsTSGHh1LcxP8a4A%3D%3D)

**Portfolio:** [Portfolio website](https://jimi121.github.io/)
