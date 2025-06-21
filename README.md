# 🎬 Movie Recommender App

A powerful movie recommendation system featuring **Content-Based Filtering**, **Collaborative Filtering**, and a **Hybrid Recommendation Engine** — all wrapped in a sleek, interactive **Streamlit** app. Designed to provide accurate and engaging suggestions based on both user behavior and movie metadata.

<!-- [**Live Demo**](https://your-streamlit-app-url)  -->

---

## About the Project

This project explores multiple recommendation strategies to suggest movies based on user interest or selection. It goes beyond basic filtering to implement and compare:

* **Content-Based Filtering** using movie metadata (genres, keywords, overview).
* **Collaborative Filtering** via latent factor models (e.g., SVD).
* **Hybrid System** that intelligently combines both for improved recommendations.

### What I Built

* A modular recommendation pipeline supporting three industry-grade methods.
* **TF-IDF**, **cosine similarity**, and **stemmed NLP features** for content-based logic.
* **Matrix factorization with Truncated SVD** for collaborative filtering.
* Dynamic **hybrid engine** that blends scores from content and collaborative systems.
* Clean, user-friendly **Streamlit interface** for live interaction.
* **Deployed on the web** with zero-install access.

---

## Recommendation Approaches

### 1. Content-Based Filtering

* Uses metadata like genres, plot keywords, and summaries.
* NLP techniques: tokenization, stemming, vectorization via TF-IDF/CountVectorizer.
* Computes **cosine similarity** to find similar movies.

### 2. Collaborative Filtering

* Learns latent user preferences via **TruncatedSVD** (similar to matrix factorization).
* Suggests movies based on similarity in user rating patterns.

### 3. Hybrid Recommendation System

* Combines both systems by **weighted scoring**.
* Aims to reduce cold-start problems and improve accuracy.

---

## Tech Stack

| Category      | Tools Used                                     |
| ------------- | ---------------------------------------------- |
| Frontend      | Streamlit                                      |
| Data Handling | Pandas, NumPy                                  |
| NLP/Modeling  | Scikit-learn, NLTK (SnowballStemmer)           |
| Visualization | Plotly                                         |
| Deployment    | Streamlit Cloud / Render / Hugging Face Spaces |

---

## Key Achievements

* Built 3 recommendation engines and integrated them seamlessly.
* Applied NLP, dimensionality reduction, and similarity metrics end-to-end.
* Created a performant app ready for production and user testing.
* Deployed a machine learning system to the web with full accessibility.
* Practiced full-stack ML development with frontend + backend integration.

---

## Run Locally

```bash
git clone https://github.com/realaryagupta/Movie-Recommender-System.git
cd Movie-Recommender-System
pip install -r requirements.txt
streamlit run main.py
```

---

## Sample Use Cases

* “Find me movies similar to *Inception*” (Content-Based)
* “What would users like me watch next?” (Collaborative)
* “Best of both worlds?” (Hybrid)

---

## Let's Connect!

* [LinkedIn](https://www.linkedin.com/in/arya-gupta-9b5873218/)
* [GitHub](https://github.com/realaryagupta)
* [aryagupta2108.ag@gmail.com](mailto:aryagupta2108.ag@gmail.com)

---

## ⭐️ Show Some Love

If you like this project, feel free to star ⭐ the repo and share your thoughts!

---