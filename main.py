import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.decomposition import TruncatedSVD
from nltk.stem.snowball import SnowballStemmer
from ast import literal_eval
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎬 CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
        font-weight: bold;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    .recommendation-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .genre-tag {
        background: #667eea;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .top-movie-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #333;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .rating-badge {
        background: #28a745;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_datasets():
    """Load and preprocess both movies metadata and ratings datasets"""
    try:
        # Load movies metadata
        movies_df = pd.read_csv('movies_metadata.csv', low_memory=False)
        
        # Load ratings data
        ratings_df = pd.read_csv('ratings_small.csv')
        
        # Clean movies data
        movies_df = movies_df.dropna(subset=['title', 'overview'])
        
        # Convert movieId to numeric, handling errors
        movies_df['id'] = pd.to_numeric(movies_df['id'], errors='coerce')
        movies_df = movies_df.dropna(subset=['id'])
        movies_df['id'] = movies_df['id'].astype(int)
        
        # Handle genres - convert from string representation of list to actual list
        def safe_literal_eval(x):
            try:
                if pd.isna(x) or x == '[]' or x == '' or str(x) == 'nan':
                    return []
                result = literal_eval(str(x))
                if isinstance(result, list):
                    return [item['name'] if isinstance(item, dict) and 'name' in item else str(item) for item in result]
                return []
            except:
                return []
        
        movies_df['genres'] = movies_df['genres'].apply(safe_literal_eval)
        
        # Clean numeric columns
        movies_df['vote_average'] = pd.to_numeric(movies_df['vote_average'], errors='coerce').fillna(0)
        movies_df['vote_count'] = pd.to_numeric(movies_df['vote_count'], errors='coerce').fillna(0)
        movies_df['popularity'] = pd.to_numeric(movies_df['popularity'], errors='coerce').fillna(0)
        movies_df['budget'] = pd.to_numeric(movies_df['budget'], errors='coerce').fillna(0)
        movies_df['revenue'] = pd.to_numeric(movies_df['revenue'], errors='coerce').fillna(0)
        
        # Extract year from release_date
        movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], errors='coerce')
        movies_df['year'] = movies_df['release_date'].dt.year
        
        # Fill missing values
        movies_df['overview'] = movies_df['overview'].fillna('')
        movies_df['tagline'] = movies_df['tagline'].fillna('')
        
        # Create combined description for content-based filtering
        movies_df['description'] = movies_df['overview'] + ' ' + movies_df['tagline']
        
        # Process ratings data
        ratings_df['movieId'] = pd.to_numeric(ratings_df['movieId'], errors='coerce')
        ratings_df['rating'] = pd.to_numeric(ratings_df['rating'], errors='coerce')
        ratings_df = ratings_df.dropna(subset=['movieId', 'rating'])
        
        # Calculate additional rating statistics
        rating_stats = ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count', 'std']
        }).round(2)
        
        rating_stats.columns = ['avg_user_rating', 'user_rating_count', 'rating_std']
        rating_stats = rating_stats.reset_index()
        
        # Merge with movies data
        movies_enhanced = movies_df.merge(rating_stats, left_on='id', right_on='movieId', how='left')
        
        # Fill NaN values for movies without ratings
        movies_enhanced['avg_user_rating'] = movies_enhanced['avg_user_rating'].fillna(0)
        movies_enhanced['user_rating_count'] = movies_enhanced['user_rating_count'].fillna(0)
        movies_enhanced['rating_std'] = movies_enhanced['rating_std'].fillna(0)
        
        # Filter out movies with very few interactions for quality
        min_ratings = max(1, int(movies_enhanced['user_rating_count'].quantile(0.1)))
        quality_movies = movies_enhanced[
            (movies_enhanced['vote_count'] >= 5) | 
            (movies_enhanced['user_rating_count'] >= min_ratings)
        ].copy()
        
        # Reset index
        quality_movies = quality_movies.reset_index(drop=True)
        
        return quality_movies, ratings_df
        
    except FileNotFoundError as e:
        st.error(f"❌ File not found: {str(e)}. Please ensure both 'movies_metadata.csv' and 'ratings_small.csv' are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None

@st.cache_data
def calculate_hybrid_score(df, content_weight=0.4, popularity_weight=0.3, user_rating_weight=0.3):
    """Calculate hybrid scores combining content, popularity, and user ratings"""
    df_scored = df.copy()
    
    # Normalize scores to 0-1 scale
    def normalize_score(series):
        if series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min())
    
    # Content score (based on vote_average)
    content_score = normalize_score(df_scored['vote_average'])
    
    # Popularity score
    popularity_score = normalize_score(df_scored['popularity'])
    
    # User rating score (from ratings_small.csv)
    user_rating_score = normalize_score(df_scored['avg_user_rating'])
    
    # Combined hybrid score
    df_scored['hybrid_score'] = (
        content_weight * content_score + 
        popularity_weight * popularity_score + 
        user_rating_weight * user_rating_score
    )
    
    return df_scored

@st.cache_data
def build_content_recommender(df):
    """Build content-based recommender using TF-IDF on movie descriptions"""
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english',
        max_features=15000  # Increased for better recommendations
    )
    
    tfidf_matrix = tfidf.fit_transform(df['description'])
    
    # Calculate cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Create indices mapping
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    return cosine_sim, indices, tfidf

def get_content_recommendations(title, cosine_sim, indices, df, num_recommendations=10):
    """Get content-based recommendations for a given movie"""
    try:
        # Ensure the title exists in indices
        if title not in indices:
            st.error(f"Movie '{title}' not found in the database.")
            return pd.DataFrame()
            
        idx = indices[title]
        
        # Get pairwise similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get scores of the most similar movies
        sim_scores = sim_scores[1:num_recommendations+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommendations
        recommendations = df.iloc[movie_indices].copy()
        
        # Add similarity score (ensure this column doesn't contain lists)
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations
        
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def build_collaborative_recommender(ratings_df, movies_df):
    """Build collaborative filtering recommender using matrix factorization"""
    try:
        # Create user-movie matrix
        user_movie_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        
        # Use SVD for dimensionality reduction
        svd = TruncatedSVD(n_components=50, random_state=42)
        user_factors = svd.fit_transform(user_movie_matrix)
        movie_factors = svd.components_.T
        
        # Calculate movie-movie similarity
        movie_similarity = cosine_similarity(movie_factors)
        
        # Create movie index mapping
        movie_ids = user_movie_matrix.columns.tolist()
        movie_idx_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        
        return movie_similarity, movie_idx_map, movie_ids
    
    except Exception as e:
        st.warning(f"Could not build collaborative filter: {str(e)}. Using content-based only.")
        return None, None, None

def get_hybrid_recommendations(title, content_sim, content_indices, collab_sim, collab_map, df, 
                             content_weight=0.7, collab_weight=0.3, num_recommendations=10):
    """Get hybrid recommendations combining content and collaborative filtering"""
    try:
        # Get content-based recommendations
        content_recs = get_content_recommendations(title, content_sim, content_indices, df, num_recommendations*2)
        
        if collab_sim is not None and collab_map is not None:
            # Get movie ID for collaborative filtering
            movie_info = df[df['title'] == title].iloc[0]
            movie_id = movie_info['id']
            
            if movie_id in collab_map:
                movie_idx = collab_map[movie_id]
                
                # Get collaborative similarities
                collab_scores = list(enumerate(collab_sim[movie_idx]))
                collab_scores = sorted(collab_scores, key=lambda x: x[1], reverse=True)
                
                # Combine scores
                final_scores = {}
                
                for _, row in content_recs.iterrows():
                    content_score = row['similarity_score']
                    movie_id_rec = row['id']
                    
                    if movie_id_rec in collab_map:
                        collab_idx = collab_map[movie_id_rec]
                        collab_score = collab_sim[movie_idx][collab_idx]
                    else:
                        collab_score = 0
                    
                    # Weighted combination
                    hybrid_score = content_weight * content_score + collab_weight * collab_score
                    final_scores[row.name] = hybrid_score
                
                # Sort by hybrid score
                sorted_indices = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
                hybrid_recs = content_recs.loc[sorted_indices[:num_recommendations]].copy()
                hybrid_recs['hybrid_score'] = [final_scores[idx] for idx in sorted_indices[:num_recommendations]]
                
                return hybrid_recs
        
        # Fallback to content-based only
        return content_recs.head(num_recommendations)
        
    except Exception as e:
        st.error(f"Error in hybrid recommendations: {str(e)}")
        return get_content_recommendations(title, content_sim, content_indices, df, num_recommendations)

def get_genre_based_recommendations(df, genres, num_recommendations=20):
    """Get recommendations based on preferred genres"""
    if not genres:
        return df.nlargest(num_recommendations, 'hybrid_score')
    
    # Score movies based on genre overlap
    def genre_score(movie_genres):
        if not movie_genres:
            return 0
        overlap = len(set(movie_genres) & set(genres))
        return overlap / len(set(movie_genres) | set(genres))  # Jaccard similarity
    
    df_copy = df.copy()
    df_copy['genre_score'] = df_copy['genres'].apply(genre_score)
    
    # Combine with hybrid score
    df_copy['final_score'] = 0.6 * df_copy['hybrid_score'] + 0.4 * df_copy['genre_score']
    
    return df_copy[df_copy['genre_score'] > 0].nlargest(num_recommendations, 'final_score')

# Main App
def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #667eea; font-size: 3rem; margin-bottom: 0.5rem;'>🎬 CineMatch Pro</h1>
        <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>Advanced Movie Recommendation Engine with Hybrid AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    if 'movies_df' not in st.session_state:
        with st.spinner("🎬 Loading movie database and building AI models... This may take a moment."):
            movies_df, ratings_df = load_datasets()
            if movies_df is None or ratings_df is None:
                st.stop()
            
            # Calculate hybrid scores
            movies_df = calculate_hybrid_score(movies_df)
            
            st.session_state.movies_df = movies_df
            st.session_state.ratings_df = ratings_df
            
            # Build recommender systems
            with st.spinner("🔧 Building recommendation engines..."):
                # Content-based recommender
                cosine_sim, indices, tfidf = build_content_recommender(movies_df)
                st.session_state.cosine_sim = cosine_sim
                st.session_state.indices = indices
                st.session_state.tfidf = tfidf
                
                # Collaborative filtering recommender
                collab_sim, collab_map, movie_ids = build_collaborative_recommender(ratings_df, movies_df)
                st.session_state.collab_sim = collab_sim
                st.session_state.collab_map = collab_map
                st.session_state.movie_ids = movie_ids
    
    movies_df = st.session_state.movies_df
    ratings_df = st.session_state.ratings_df
    cosine_sim = st.session_state.cosine_sim
    indices = st.session_state.indices
    collab_sim = st.session_state.collab_sim
    collab_map = st.session_state.collab_map
    
    # Get all unique genres from the dataset
    all_genres = list(set([genre for sublist in movies_df['genres'] for genre in sublist if genre]))
    all_genres = sorted([g for g in all_genres if g and str(g) != 'nan'])
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Recommendation Settings")
        
        # Recommendation type
        rec_type = st.selectbox(
            "Recommendation Method:",
            ["🤖 Hybrid AI (Best)", "📝 Content-Based", "🎭 Genre-Based", "📊 Popular Movies"]
        )
        
        # Movie selection for recommendations
        st.markdown("#### Select a Movie")
        
        # Search functionality
        search_term = st.text_input("🔍 Search for a movie:", placeholder="Type movie name...")
        
        if search_term:
            matching_movies = movies_df[movies_df['title'].str.contains(search_term, case=False, na=False)]['title'].tolist()
            if matching_movies:
                selected_movie = st.selectbox(
                    "Choose from search results:",
                    options=matching_movies[:50]  # Limit for performance
                )
            else:
                st.warning("No movies found matching your search.")
                selected_movie = st.selectbox(
                    "Or choose from top-rated movies:",
                    options=movies_df.nlargest(100, 'hybrid_score')['title'].tolist()
                )
        else:
            # Show top movies by default
            top_movies = movies_df.nlargest(100, 'hybrid_score')['title'].tolist()
            selected_movie = st.selectbox(
                "Choose from top-rated movies:",
                options=top_movies
            )
        
        # Number of recommendations
        num_recs = st.slider("Number of recommendations:", 5, 25, 12)
        
        # Genre preferences for genre-based recommendations
        if rec_type == "🎭 Genre-Based":
            st.markdown("#### Genre Preferences")
            selected_genres = st.multiselect(
                "Select your favorite genres:",
                options=all_genres,
                default=all_genres[:3] if len(all_genres) >= 3 else all_genres
            )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            if rec_type == "🤖 Hybrid AI (Best)":
                content_weight = st.slider("Content Weight:", 0.0, 1.0, 0.7, 0.1)
                collab_weight = 1.0 - content_weight
                st.write(f"Collaborative Weight: {collab_weight:.1f}")
            
            min_rating = st.slider("Minimum Rating:", 0.0, 10.0, 0.0, 0.5)
            min_votes = st.slider("Minimum Vote Count:", 0, 1000, 0, 50)
        
        # Database statistics
        st.markdown("### 📊 Database Stats")
        
        total_movies = len(movies_df)
        total_ratings = len(ratings_df)
        unique_users = ratings_df['userId'].nunique()
        avg_rating = ratings_df['rating'].mean()
        
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{total_movies:,}</h3>
                <p>Movies</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{len(all_genres)}</h3>
                <p>Genres</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stats_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{total_ratings:,}</h3>
                <p>Ratings</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3>{unique_users:,}</h3>
                <p>Users</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Get Recommendations", "🏆 Top Movies", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.markdown("## 🎯 AI-Powered Movie Recommendations")
        
        # Filter movies based on advanced settings
        filtered_df = movies_df[
            (movies_df['vote_average'] >= min_rating) & 
            (movies_df['vote_count'] >= min_votes)
        ].copy()
        
        if rec_type == "🎭 Genre-Based":
            # Genre-based recommendations
            with st.spinner("🎭 Finding movies based on your genre preferences..."):
                recommendations = get_genre_based_recommendations(filtered_df, selected_genres, num_recs)
            
            st.markdown(f"### 🎬 Recommended {', '.join(selected_genres)} Movies")
        
        elif rec_type == "📊 Popular Movies":
            # Popular movies
            recommendations = filtered_df.nlargest(num_recs, 'hybrid_score')
            st.markdown("### 🌟 Most Popular Movies")
        
        elif selected_movie:
            # Display selected movie info
            movie_info = movies_df[movies_df['title'] == selected_movie].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class='movie-card'>
                    <h3>{movie_info['title']}</h3>
                    <p><strong>Year:</strong> {int(movie_info['year']) if pd.notna(movie_info['year']) else 'Unknown'}</p>
                    <p><strong>IMDb Rating:</strong> ⭐ {movie_info['vote_average']:.1f}/10</p>
                    <p><strong>User Rating:</strong> 🎭 {movie_info['avg_user_rating']:.1f}/5.0</p>
                    <p><strong>Votes:</strong> {int(movie_info['vote_count']):,}</p>
                    <p><strong>User Ratings:</strong> {int(movie_info['user_rating_count']):,}</p>
                    <p><strong>Popularity:</strong> {movie_info['popularity']:.1f}</p>
                    <p><strong>Hybrid Score:</strong> {movie_info['hybrid_score']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Genre tags
                if movie_info['genres']:
                    st.markdown("**Genres:**")
                    genre_html = ""
                    for genre in movie_info['genres']:
                        genre_html += f"<span class='genre-tag'>{genre}</span>"
                    st.markdown(genre_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Plot Overview:**")
                overview_text = movie_info['overview'][:500] + "..." if len(movie_info['overview']) > 500 else movie_info['overview']
                st.write(overview_text if overview_text else "No overview available.")
                
                if movie_info['tagline']:
                    st.markdown(f"*\"{movie_info['tagline']}\"*")
            
            st.markdown("---")
            
            # Get recommendations based on selected method
            if rec_type == "🤖 Hybrid AI (Best)":
                with st.spinner(f"🤖 AI is analyzing '{selected_movie}' and finding similar movies..."):
                    recommendations = get_hybrid_recommendations(
                        selected_movie, cosine_sim, indices, collab_sim, collab_map, 
                        filtered_df, content_weight, collab_weight, num_recs
                    )
                st.markdown(f"## 🤖 AI Hybrid Recommendations for '{selected_movie}'")
            
            elif rec_type == "📝 Content-Based":
                with st.spinner(f"📝 Analyzing content similarity to '{selected_movie}'..."):
                    recommendations = get_content_recommendations(selected_movie, cosine_sim, indices, filtered_df, num_recs)
                st.markdown(f"## 📝 Content-Based Recommendations for '{selected_movie}'")
        
        # Display recommendations
        if 'recommendations' in locals() and not recommendations.empty:
            for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 6, 2])
                    
                    with col1:
                        st.markdown(f"### #{idx}")
                    
                    with col2:
                        st.markdown(f"**{movie['title']}** ({int(movie['year']) if pd.notna(movie['year']) else 'Unknown'})")
                        
                        # Overview
                        overview = movie['overview'][:200] + "..." if len(movie['overview']) > 200 else movie['overview']
                        st.write(overview if overview else "No overview available.")
                        
                        # Genre tags
                        if movie['genres']:
                            genre_html = ""
                            for genre in movie['genres']:
                                genre_html += f"<span class='genre-tag'>{genre}</span>"
                            st.markdown(genre_html, unsafe_allow_html=True)
                    
                    with col3:
                        st.metric("IMDb Rating", f"{movie['vote_average']:.1f}/10")
                        if movie['avg_user_rating'] > 0:
                            st.metric("User Rating", f"{movie['avg_user_rating']:.1f}/5.0")
                        
                        # Show similarity or hybrid score
                        if 'similarity_score' in movie:
                            st.metric("Similarity", f"{movie['similarity_score']:.3f}")
                        elif 'hybrid_score' in movie:
                            st.metric("AI Score", f"{movie['hybrid_score']:.3f}")
                    
                    st.markdown("---")
        else:
            st.info("No recommendations found with the current filters. Try adjusting your settings.")
    
    with tab2:
        st.markdown("## 🏆 Top Movies Rankings")
        
        ranking_type = st.selectbox(
            "Ranking Method:",
            ["🤖 Hybrid AI Score", "⭐ IMDb Rating", "🎭 User Rating", "📈 Popularity", "🎬 By Genre"]
        )
        
        if ranking_type == "🎬 By Genre":
            genre_filter = st.selectbox("Select Genre:", ["All"] + all_genres)
            if genre_filter != "All":
                genre_movies = movies_df[movies_df['genres'].apply(lambda x: genre_filter in x if isinstance(x, list) else False)]
                top_movies = genre_movies.nlargest(25, 'hybrid_score')
                st.markdown(f"### 🎭 Top {genre_filter} Movies")
            else:
                top_movies = movies_df.nlargest(25, 'hybrid_score')
                st.markdown("### 🌟 Top Movies (All Genres)")
        else:
            # Determine sorting column
            if ranking_type == "🤖 Hybrid AI Score":
                sort_col = 'hybrid_score'
                st.markdown("### 🤖 Top Movies by AI Hybrid Score")
            elif ranking_type == "⭐ IMDb Rating":
                sort_col = 'vote_average'
                st.markdown("### ⭐ Highest Rated Movies (IMDb)")
            elif ranking_type == "🎭 User Rating":
                sort_col = 'avg_user_rating'
                movies_df_filtered = movies_df[movies_df['user_rating_count'] >= 10]  # Minimum user ratings
                top_movies = movies_df_filtered.nlargest(25, sort_col)
                st.markdown("### 🎭 Highest Rated Movies (Users)")
            else:  # Popularity
                sort_col = 'popularity'
                st.markdown("### 📈 Most Popular Movies")
            
            if ranking_type != "🎭 User Rating":
                top_movies = movies_df.nlargest(25, sort_col)
        
        # Display top movies
        for idx, (_, movie) in enumerate(top_movies.iterrows(), 1):
            score_display = ""
            if ranking_type == "🤖 Hybrid AI Score":
                score_display = f"{movie['hybrid_score']:.3f}"
            elif ranking_type == "⭐ IMDb Rating":
                score_display = f"{movie['vote_average']:.1f}/10"
            elif ranking_type == "🎭 User Rating":
                score_display = f"{movie['avg_user_rating']:.1f}/5.0"
            elif ranking_type == "📈 Popularity":
                score_display = f"{movie['popularity']:.1f}"
            else:
                score_display = f"{movie['hybrid_score']:.3f}"
            
            st.markdown(f"""
            <div class='top-movie-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='flex: 1;'>
                        <h4>#{idx}. {movie['title']} ({int(movie['year']) if pd.notna(movie['year']) else 'Unknown'})</h4>
                        <p><strong>IMDb:</strong> ⭐ {movie['vote_average']:.1f}/10 | <strong>Votes:</strong> {int(movie['vote_count']):,}</p>
                        {f"<p><strong>User Rating:</strong> 🎭 {movie['avg_user_rating']:.1f}/5.0 | <strong>User Votes:</strong> {int(movie['user_rating_count']):,}</p>" if movie['avg_user_rating'] > 0 else ""}
                        <p><strong>Popularity:</strong> {movie['popularity']:.1f}</p>
                    </div>
                    <div style='text-align: right;'>
                        <h3 class='rating-badge'>{score_display}</h3>
                        <small>{ranking_type.split()[1] if len(ranking_type.split()) > 1 else 'Score'}</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Genre tags
            if isinstance(movie['genres'], list) and movie['genres']:
                genre_html = ""
                for genre in movie['genres']:
                    genre_html += f"<span class='genre-tag'>{genre}</span>"
                st.markdown(genre_html, unsafe_allow_html=True)
            
            # Show brief overview
            if movie['overview']:
                overview = movie['overview'][:150] + "..." if len(movie['overview']) > 150 else movie['overview']
                st.markdown(f"*{overview}*")
            
            st.markdown("<br>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## 📊 Advanced Movie Analytics")
        
        # Genre distribution
        genre_counts = {}
        for genres in movies_df['genres']:
            if isinstance(genres, list):
                for genre in genres:
                    if genre and str(genre) != 'nan':
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎭 Genre Distribution")
            if genre_counts:
                genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Count'])
                genre_df = genre_df.sort_values('Count', ascending=True).tail(15)
                fig_genre = px.bar(genre_df, x='Count', y='Genre', orientation='h',
                                 color='Count', color_continuous_scale='viridis',
                                 title="Number of Movies by Genre")
                fig_genre.update_layout(height=500)
                st.plotly_chart(fig_genre, use_container_width=True)
        
        with col2:
            st.markdown("### ⭐ Rating Distribution Comparison")
            
            # Compare IMDb ratings vs User ratings
            fig_rating = make_subplots(rows=2, cols=1, 
                                     subplot_titles=('IMDb Ratings', 'User Ratings'),
                                     vertical_spacing=0.1)
            
            # IMDb ratings
            fig_rating.add_trace(
                go.Histogram(x=movies_df[movies_df['vote_average'] > 0]['vote_average'], 
                           nbinsx=20, name='IMDb Ratings', marker_color='blue'),
                row=1, col=1
            )
            
            # User ratings (scaled to 10 for comparison)
            user_ratings_scaled = movies_df[movies_df['avg_user_rating'] > 0]['avg_user_rating'] * 2
            fig_rating.add_trace(
                go.Histogram(x=user_ratings_scaled, 
                           nbinsx=20, name='User Ratings (scaled)', marker_color='red'),
                row=2, col=1
            )
            
            fig_rating.update_layout(height=500, showlegend=False)
            fig_rating.update_xaxes(title_text="Rating", row=2, col=1)
            fig_rating.update_yaxes(title_text="Count")
            st.plotly_chart(fig_rating, use_container_width=True)
        
        # Advanced analytics
        st.markdown("### 📅 Movies and Ratings Over Time")
        
        # Year-wise analysis
        year_data = movies_df[movies_df['year'].notna() & (movies_df['year'] > 1900) & (movies_df['year'] <= 2024)]
        if not year_data.empty:
            yearly_stats = year_data.groupby('year').agg({
                'title': 'count',
                'vote_average': 'mean',
                'avg_user_rating': 'mean',
                'popularity': 'mean'
            }).reset_index()
            yearly_stats.columns = ['Year', 'Movie_Count', 'Avg_IMDb_Rating', 'Avg_User_Rating', 'Avg_Popularity']
            
            fig_timeline = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Movies Released by Year', 'Average IMDb Rating by Year', 
                              'Average User Rating by Year', 'Average Popularity by Year'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Movies count by year
            fig_timeline.add_trace(
                go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Movie_Count'],
                         mode='lines+markers', name='Movies Released', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Average ratings by year
            fig_timeline.add_trace(
                go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Avg_IMDb_Rating'],
                         mode='lines+markers', name='IMDb Rating', line=dict(color='green')),
                row=1, col=2
            )
            
            # User ratings by year
            fig_timeline.add_trace(
                go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Avg_User_Rating'],
                         mode='lines+markers', name='User Rating', line=dict(color='red')),
                row=2, col=1
            )
            
            # Popularity by year
            fig_timeline.add_trace(
                go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Avg_Popularity'],
                         mode='lines+markers', name='Popularity', line=dict(color='purple')),
                row=2, col=2
            )
            
            fig_timeline.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Rating correlation analysis
        st.markdown("### 🔗 Rating Correlations")
        
        correlation_data = movies_df[
            (movies_df['vote_average'] > 0) & 
            (movies_df['avg_user_rating'] > 0) & 
            (movies_df['user_rating_count'] >= 5)
        ]
        
        if not correlation_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # IMDb vs User ratings scatter plot
                fig_corr = px.scatter(
                    correlation_data, 
                    x='vote_average', 
                    y='avg_user_rating',
                    size='user_rating_count',
                    hover_data=['title', 'year'],
                    title='IMDb Rating vs User Rating',
                    labels={'vote_average': 'IMDb Rating (0-10)', 'avg_user_rating': 'User Rating (0-5)'}
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                # Correlation statistics
                corr_imdb_user = correlation_data['vote_average'].corr(correlation_data['avg_user_rating'])
                corr_popularity_rating = correlation_data['popularity'].corr(correlation_data['vote_average'])
                corr_votes_rating = correlation_data['vote_count'].corr(correlation_data['vote_average'])
                
                st.markdown(f"""
                **Correlation Analysis:**
                - IMDb vs User Rating: **{corr_imdb_user:.3f}**
                - Popularity vs IMDb Rating: **{corr_popularity_rating:.3f}**
                - Vote Count vs IMDb Rating: **{corr_votes_rating:.3f}**
                
                *Correlation ranges from -1 (negative) to +1 (positive)*
                """)
                
                # Top performing movies by different metrics
                st.markdown("#### 🏆 Top Performers")
                
                # Highest budget vs revenue
                budget_movies = movies_df[movies_df['budget'] > 1000000].nlargest(5, 'revenue')
                if not budget_movies.empty:
                    st.markdown("**Highest Revenue Movies:**")
                    for _, movie in budget_movies.iterrows():
                        roi = (movie['revenue'] - movie['budget']) / movie['budget'] * 100 if movie['budget'] > 0 else 0
                        st.write(f"• {movie['title']} - Revenue: ${movie['revenue']:,.0f}M (ROI: {roi:.1f}%)")
        
        # User engagement analysis
        st.markdown("### 👥 User Engagement Analysis")
        
        if not ratings_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating distribution
                rating_dist = ratings_df['rating'].value_counts().sort_index()
                fig_rating_dist = px.bar(
                    x=rating_dist.index, 
                    y=rating_dist.values,
                    title='User Rating Distribution',
                    labels={'x': 'Rating', 'y': 'Count'}
                )
                st.plotly_chart(fig_rating_dist, use_container_width=True)
            
            with col2:
                # User activity
                user_activity = ratings_df.groupby('userId').size().describe()
                st.markdown("**User Activity Statistics:**")
                st.write(f"• Total Users: {ratings_df['userId'].nunique():,}")
                st.write(f"• Average Ratings per User: {user_activity['mean']:.1f}")
                st.write(f"• Most Active User: {user_activity['max']:.0f} ratings")
                st.write(f"• Median Ratings per User: {user_activity['50%']:.1f}")
                
                # Most rated movies
                most_rated = ratings_df['movieId'].value_counts().head(10)
                movie_titles = []
                for movie_id in most_rated.index:
                    movie_row = movies_df[movies_df['id'] == movie_id]
                    if not movie_row.empty:
                        movie_titles.append(movie_row.iloc[0]['title'])
                    else:
                        movie_titles.append(f"Movie ID {movie_id}")
                
                st.markdown("**Most Rated Movies:**")
                for i, (movie_id, count) in enumerate(most_rated.items()):
                    st.write(f"{i+1}. {movie_titles[i]} - {count:,} ratings")
    
    with tab4:
        st.markdown("## ℹ️ About CineMatch Pro")
        
        # Dynamic statistics
        total_movies = len(movies_df)
        total_ratings = len(ratings_df)
        unique_users = ratings_df['userId'].nunique()
        avg_imdb_rating = movies_df['vote_average'].mean()
        avg_user_rating = movies_df[movies_df['avg_user_rating'] > 0]['avg_user_rating'].mean()
        
        st.markdown(f"""
        ### 🎬 What is CineMatch Pro?
        
        CineMatch Pro is an advanced AI-powered movie recommendation system that combines multiple 
        machine learning techniques to provide personalized movie suggestions. It analyzes your 
        preferences using both content analysis and collaborative filtering to find movies you'll love.
        
        ### 🤖 Advanced AI Techniques
        
        **1. Hybrid Recommendation System:**
        - **Content-Based Filtering:** Analyzes movie plots, genres, and metadata using TF-IDF vectorization
        - **Collaborative Filtering:** Uses Matrix Factorization (SVD) to find patterns in user ratings
        - **Hybrid Scoring:** Intelligently combines both approaches for superior recommendations
        
        **2. Smart Preprocessing:**
        - Dynamic genre extraction from structured data
        - Multi-source rating aggregation (IMDb + User ratings)
        - Intelligent data cleaning and normalization
        - Real-time similarity calculations
        
        **3. Advanced Analytics:**
        - Correlation analysis between different rating systems
        - Temporal trend analysis of movies and ratings
        - User engagement pattern analysis
        - Genre popularity tracking
        
        ### 📊 Current Database Statistics
        
        - **Total Movies in Database:** {total_movies:,}
        - **Total User Ratings:** {total_ratings:,}
        - **Unique Users:** {unique_users:,}
        - **Average IMDb Rating:** {avg_imdb_rating:.2f}/10
        - **Average User Rating:** {avg_user_rating:.2f}/5.0
        - **Unique Genres:** {len(all_genres)}
        - **Rating Coverage:** {(movies_df['avg_user_rating'] > 0).sum():,} movies have user ratings
        
        ### 🎯 Recommendation Methods
        
        **🤖 Hybrid AI (Recommended):**
        - Combines content similarity with collaborative filtering
        - Adjustable weights for personalization
        - Best overall accuracy and diversity
        
        **📝 Content-Based:**
        - Pure content similarity using TF-IDF on movie descriptions
        - Great for finding movies with similar plots/themes
        - Works well for new or less-rated movies
        
        **🎭 Genre-Based:**
        - Recommendations based on your favorite genres
        - Uses Jaccard similarity for genre matching
        - Perfect for exploring specific genres
        
        **📊 Popular Movies:**
        - Hybrid scoring combining popularity, ratings, and user engagement
        - Great for discovering critically acclaimed films
        - Balanced approach considering multiple quality metrics
        
        ### 💡 Pro Tips for Best Results
        
        1. **Try Different Methods:** Each recommendation type offers unique perspectives
        2. **Adjust Weights:** In Hybrid mode, experiment with content vs collaborative weights
        3. **Use Filters:** Set minimum ratings and vote counts for quality control
        4. **Explore Genres:** Use genre-based recommendations to discover new categories
        5. **Check Analytics:** Understand patterns in your movie preferences
        6. **Search Functionality:** Use the search to find specific movies as starting points
        
        ### 🔬 Technical Implementation
        
        **Machine Learning Libraries:**
        - **Scikit-learn:** TF-IDF vectorization, cosine similarity, SVD
        - **Pandas/NumPy:** Data manipulation and numerical computations
        - **Plotly:** Interactive visualizations and analytics
        
        **Key Algorithms:**
        - **TF-IDF (Term Frequency-Inverse Document Frequency):** Content analysis
        - **Cosine Similarity:** Measuring content similarity between movies
        - **SVD (Singular Value Decomposition):** Matrix factorization for collaborative filtering
        - **Hybrid Scoring:** Weighted combination of multiple recommendation signals
        
        **Performance Optimizations:**
        - **Streamlit Caching:** @st.cache_data for expensive computations
        - **Matrix Operations:** Optimized NumPy operations for similarity calculations
        - **Smart Filtering:** Efficient data preprocessing and filtering
        
        ### 📈 Future Enhancements
        
        - **Deep Learning Integration:** Neural collaborative filtering
        - **Real-time Learning:** Adaptive recommendations based on user feedback
        - **Multi-modal Analysis:** Integration of movie posters, trailers, and cast information
        - **Temporal Dynamics:** Time-aware recommendations considering changing preferences
        - **Cross-platform Integration:** API connectivity with streaming services
        - **Advanced NLP:** Sentiment analysis of reviews and plot summaries
        
        ### 📚 Data Sources
        
        - **movies_metadata.csv:** Comprehensive movie metadata including plots, genres, ratings
        - **ratings_small.csv:** User rating data for collaborative filtering
        - **Dynamic Processing:** Real-time computation of hybrid scores and similarities
        
        ---
        
        **Built with:** Streamlit, Scikit-learn, Pandas, NumPy, Plotly
        
        **Recommendation Engine:** Hybrid AI combining Content-Based + Collaborative Filtering
        
        **Last Updated:** Real-time processing of your uploaded datasets
        """)
        
        # Performance metrics
        st.markdown("### ⚡ System Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Content Similarity Matrix", 
                f"{len(movies_df)} × {len(movies_df)}", 
                "Computed in memory"
            )
        
        with col2:
            if collab_sim is not None:
                st.metric(
                    "Collaborative Filter", 
                    "✅ Active", 
                    f"SVD with 50 components"
                )
            else:
                st.metric(
                    "Collaborative Filter", 
                    "⚠️ Limited", 
                    "Using content-based only"
                )
        
        with col3:
            hybrid_coverage = (movies_df['hybrid_score'] > 0).sum()
            st.metric(
                "Hybrid Score Coverage", 
                f"{hybrid_coverage:,}/{total_movies:,}", 
                f"{(hybrid_coverage/total_movies)*100:.1f}%"
            )

if __name__ == "__main__":
    main()