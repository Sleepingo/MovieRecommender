import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import difflib

st.set_page_config(page_title="Movie Recommender System", layout="wide")

st.markdown("""
    <style>
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
    }
    p, div, span, label {
        color: #000000 !important;
    }
    .stMarkdown {
        color: #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender System")
st.write("A group project combining Content-Based, Collaborative, and Hybrid Filtering")

tab1, tab2, tab3 = st.tabs(["Content-Based", "Collaborative", "Hybrid"])

# ============================================================
# TAB 1: CONTENT-BASED FILTERING (Ping)
# ============================================================
with tab1:
    st.header("Content-Based Filtering (TF-IDF + Cosine Similarity)")

    @st.cache_data
    def load_data():
        df = pd.read_csv('data/tmdb_5000_movies.csv')

        def extract_names(text):
            try:
                items = ast.literal_eval(text)
                return ", ".join([i['name'] for i in items])
            except:
                return ""

        df['genres_clean'] = df['genres'].apply(extract_names)
        df['keywords_clean'] = df['keywords'].apply(extract_names)
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.astype('Int64')
        df['overview'] = df['overview'].fillna('')
        df['content'] = df['genres_clean'] + ' ' + df['keywords_clean'] + ' ' + df['overview']
        return df

    @st.cache_data
    def compute_similarity(df):
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['content'])
        return cosine_similarity(tfidf_matrix, tfidf_matrix)

    df = load_data()
    cosine_sim = compute_similarity(df)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    def get_input_features(title):
        idx = indices[title]
        movie = df.loc[idx]
        keywords_display = movie['keywords_clean']
        if len(keywords_display) > 60:
            keywords_display = keywords_display[:60] + "..."
        return pd.DataFrame({
            "title": [movie['title']],
            "year": [movie['year']],
            "genres": [movie['genres_clean']],
            "keywords": [keywords_display],
            "vote_average": [movie['vote_average']]
        })

    def recommend(title, top_n=10):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        scores = [round(s[1], 4) for s in sim_scores]

        result = df[['title', 'year', 'genres_clean', 'keywords_clean', 'vote_average']].iloc[movie_indices].copy()
        result.insert(0, 'rank', range(1, len(result) + 1))
        result['similarity_score'] = scores

       
        result['keywords_clean'] = result['keywords_clean'].apply(
            lambda x: ', '.join(x.split(', ')[:5]) if x else ''
        )

        result = result.rename(columns={
            'title': 'movie_title',
            'genres_clean': 'genres',
            'keywords_clean': 'keywords',
            'vote_average': 'rating'
        })
        return result.reset_index(drop=True)

    # ---------- UI ----------
    movie_list = sorted(df['title'].dropna().unique())

    input_method = st.radio("Select input method:", ["Choose from list", "Type movie name"], horizontal=True, key="cb_input_method")

    if input_method == "Choose from list":
        selected_movie = st.selectbox("Choose a movie:", movie_list, key="cb_selectbox")
    else:
        user_input = st.text_input("Type a movie name (in English, more complete = more accurate):", key="cb_text_input")
        selected_movie = None

        if user_input:
            exact_match = [m for m in movie_list if m.lower() == user_input.lower()]
            if exact_match:
                selected_movie = exact_match[0]
                st.success(f"✓ Found: {selected_movie}")
            else:
                close_matches = difflib.get_close_matches(user_input, movie_list, n=5, cutoff=0.4)
                if close_matches:
                    st.write("No exact match found. Did you mean:")
                    selected_movie = st.selectbox("Select one:", close_matches, key="cb_close_match")
                else:
                    st.warning("No matching movie found. Please try a different name.")

    top_n = st.slider("Number of recommendations:", 5, 20, 10, key="cb_slider")

    if st.button("Get Recommendations", key="cb_button") and selected_movie:
        st.subheader("Input Movie Features")
        st.table(get_input_features(selected_movie))

        st.subheader("Recommended Movies")
        results = recommend(selected_movie, top_n)
        st.dataframe(results, use_container_width=True, hide_index=True)

        avg_score = results['similarity_score'].mean()
        st.caption(f"Average similarity score of top {top_n} recommendations: {avg_score:.4f}")

# ============================================================
# TAB 2: COLLABORATIVE FILTERING (Jiun Hui)
# ============================================================
with tab2:
    st.header("Collaborative Filtering")
    st.write(
        "Movie recommendations based on user rating behaviour."
    )

    # 1. Load Collaborative Filtering Data
    @st.cache_data
    def load_collaborative_data():

        # Load Ratings
        ratings = pd.read_csv(
            "data/tmdb_movie_ratings.csv",
            usecols=[
                "userId",
                "ratingId",
                "rating"
            ]
        )

        # Limit users
        ratings = ratings[
            ratings["userId"] <= 600
        ].copy()

        # Load TMDB Movie Dataset
        movies = pd.read_csv(
            "data/tmdb_movie_dataset.csv",
            usecols=[
                "ratingId",
                "tmdbId",
                "title",
                "vote_average",
                "vote_count"
            ]
        )

        # Keep only required columns
        movies = movies.drop_duplicates(
            subset="ratingId"
        )

        merged = ratings.merge(
                movies,
                on="ratingId",
                how="inner"
            )
        merged = merged[
            [
                "userId",
                "ratingId",
                "rating",
                "tmdbId",
                "title",
                "vote_average",
                "vote_count"
            ]
        ]

        return merged
    
    data_cf = load_collaborative_data()
    print("CF Data Shape:", data_cf.shape)
    print("Number of Users:", data_cf["userId"].nunique())
    print("Number of Movies:", data_cf["title"].nunique())

    # 2. Movie Statistics
    movie_stats = (
        data_cf
        .groupby("title")["rating"]
        .agg(
            average_rating="mean",
            number_of_ratings="count"
        )
        .reset_index()
    )

    movie_stats["average_rating"] = (
        movie_stats["average_rating"]
        .round(2)
    )

    # 3. Create User-Movie Matrix
    movie_matrix = data_cf.pivot_table(
        index="userId",
        columns="title",
        values="rating"
    )

    # 4. Movie Selection
    movie_list_cf = sorted(
        movie_matrix.columns.tolist()
    )

    selected_movie_cf = st.selectbox(
        "Choose a movie:",
        movie_list_cf,
        key="cf_movie"
    )

    # 5. Number of Recommendations
    top_n_cf = st.slider(
        "Number of recommendations:",
        min_value=5,
        max_value=20,
        value=10,
        key="cf_slider"
    )

    # 6. Generate Recommendations
    if st.button(
        "Get Collaborative Recommendations",
        key="cf_button"
    ):

        # Calculate Correlation
        similar = movie_matrix.corrwith(
            movie_matrix[selected_movie_cf]
        )

        corr = pd.DataFrame(
            similar,
            columns=["Correlation"]
        )

        corr.dropna(
            inplace=True
        )

        # Remove Selected Movie
        recommendations = (
            corr
            .drop(selected_movie_cf)
        )

        # Add Number of User Ratings
        recommendations = recommendations.join(
            movie_stats.set_index("title")[
                [
                    "number_of_ratings"
                ]
            ]
        )

        # Filter Movies with Enough User Ratings
        recommendations = recommendations[
            recommendations["number_of_ratings"] >= 100
        ]

        # Sort by Correlation
        recommendations = (
            recommendations
            .sort_values(
                "Correlation",
                ascending=False
            )
        )

        # Take Top N
        recommendations = (
            recommendations
            .head(top_n_cf)
        )

        # 7. Selected Movie Information
        selected_info = data_cf[
            data_cf["title"] == selected_movie_cf
        ].iloc[0]

        selected_stats = movie_stats[
            movie_stats["title"] == selected_movie_cf
        ].iloc[0]

        st.subheader(
            "Selected Movie"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write(
                "**Movie Title**"
            )
            st.write(
                selected_movie_cf
            )

        with col2:
            st.write(
                "**Average User Rating**"
            )
            st.write(
                selected_stats["average_rating"]
            )


        with col3:
            st.write(
                "**Number of User Ratings**"
            )
            st.write(
                int(
                    selected_stats[
                        "number_of_ratings"
                    ]
                )
            )

        with col4:
            st.write(
                "**TMDB Rating**"
            )
            st.write(
                selected_info[
                    "vote_average"
                ]
            )

        # 8. Recommended Movies
        st.subheader(
            "Recommended Movies"
        )

        if recommendations.empty:
            st.warning(
                "No recommendations found."
            )
        else:
            result = (
                recommendations
                .reset_index()
            )

            # Add Rank
            result.insert(
                0,
                "Rank",
                range(
                    1,
                    len(result) + 1
                )
            )

            # Rename Columns
            result = result.rename(
                columns={
                    "title":
                        "Movie Title",

                    "Correlation":
                        "Correlation Score",

                    "number_of_ratings":
                        "Number of User Ratings"
                }
            )

            # Round Correlation Score
            result[
                "Correlation Score"
            ] = (
                result[
                    "Correlation Score"
                ].round(4)
            )

            # Display Table
            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            # 9. Average Correlation
            avg_correlation = (
                recommendations[
                    "Correlation"
                ].mean()
            )

            st.caption(
                f"Average Correlation Score "
                f"of Top {len(recommendations)} "
                f"Recommendations: "
                f"{avg_correlation:.4f}"
            )

# ============================================================
# TAB 3: HYBRID FILTERING (Shi Min)
# ============================================================
with tab3:
    st.header("Hybrid Filtering")
    st.info("🚧 Coming soon — waiting for Shi Min's recommend function.")