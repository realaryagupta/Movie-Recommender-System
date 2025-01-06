
---

# Movie Recommendation System

## Overview

Welcome to my **Movie Recommendation System** project! This repository demonstrates the creation of multiple recommendation engines for suggesting movies to users. The system leverages a combination of various algorithms, including **Simple Recommender**, **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Engine**. By using techniques like **TF-IDF**, **Cosine Similarity**, and **Singular Value Decomposition (SVD)**, I aim to provide highly personalized movie recommendations based on user preferences and movie features.

---

## Features

- **Simple Recommender:** 
  - A basic system that ranks movies based on popularity and vote averages using the IMDb weighted rating system.
  
- **Content-Based Recommender:** 
  - Two models:
    1. Movie overview and tagline-based recommendations.
    2. Metadata-based recommendations (genre, cast, crew, and keywords).
  
- **Collaborative Filtering:** 
  - Using the **Surprise** library, this engine predicts user ratings for movies and recommends them based on user preferences.

- **Hybrid Engine:** 
  - A fusion of **Content-Based** and **Collaborative Filtering** models that generates more accurate and personalized movie recommendations by combining the strengths of both approaches.

---

## Key Concepts

- **Cosine Similarity:** Used for content-based recommendations, where movie similarities are calculated based on their features.
  
- **IMDB Weighted Rating Formula:** Helps rank movies based on a combination of their average ratings and the number of votes they’ve received.
  
- **Singular Value Decomposition (SVD):** A powerful collaborative filtering technique used to predict ratings based on user-movie interactions.

---

## Getting Started

To run the project locally, follow these steps:

### Prerequisites

- Python 3.x
- Jupyter Notebook (or any other notebook environment)
- Necessary Python libraries:
  - `pandas`
  - `numpy`
  - `sklearn`
  - `surprise`
  - `nltk`
  - `matplotlib`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git](https://github.com/realaryagupta/Movie-Recommender-System.git

   ```
2. Navigate to the project directory:
   ```bash
   cd movie-recommendation-system
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter notebook and run it:
   ```bash
   jupyter notebook
   ```

---

## How It Works

1. **Simple Recommender:**
   - Uses movie vote counts and average ratings to generate top movie charts. It utilizes the **IMDB Weighted Rating formula** to calculate ratings and ranks movies accordingly.
   
2. **Content-Based Filtering:**
   - Calculates movie similarity using **TF-IDF** for overviews and taglines.
   - For metadata-based recommendations, it considers features like **genres**, **cast**, **crew**, and **keywords** to compute similarity scores between movies.

3. **Collaborative Filtering:**
   - Implements **Singular Value Decomposition (SVD)** using the **Surprise** library to predict movie ratings based on user history and preferences.
   
4. **Hybrid Engine:**
   - Combines content-based and collaborative filtering models, enhancing the recommendation quality by leveraging both the similarities of movies and user behavior.

---

## Sample Output

- **Top Movies**:
  - Top 25 movies sorted by similarity or rating
  - Genre-specific movie charts (e.g., Top Romance, Top Action)
  
- **User-Specific Recommendations**:
  - Personalized recommendations based on user movie ratings and preferences.

---

## Future Enhancements

- **Refining Hybrid Models:** Experiment with different weightings for features like genres, cast, and crew.
- **Real-time Recommendations:** Implement real-time personalized recommendations using dynamic user data.
- **Expand Dataset:** Integrate additional movie features like **plot summaries**, **release dates**, and **budget** for better recommendations.
- **User Interface:** Create a web-based interface where users can input their movie preferences and get recommendations instantly.

---

## Contributing

I welcome contributions to improve the recommender system! Feel free to fork the repo, open issues, or submit pull requests. 

### Steps to Contribute:
1. Fork the repo
2. Create a new branch
3. Make changes and commit 
4. Push to the branch 
5. Open a pull request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **TMDB Dataset**: For providing an extensive collection of movie data.
- **Surprise Library**: For making collaborative filtering easy with their powerful algorithms.

---

## Contact

For any questions or feedback, feel free to reach out via the GitHub Issues tab.

---
