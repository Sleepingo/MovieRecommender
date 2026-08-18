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
        # Use the same TMDB dataset and preprocessing as the
        # content_based_filtering.ipynb notebook.
        df = pd.read_csv("data/tmdb_movie_dataset.csv")

        def extract_names(text):
            try:
                items = ast.literal_eval(text)
                if isinstance(items, list):
                    return " ".join(
                            str(item.get("name", ""))
                            for item in items
                            if isinstance(item, dict)
)
            except (ValueError, SyntaxError, TypeError):
                pass
            return ""

        df["genres_clean"] = df["genres"].fillna("").apply(extract_names)
        df["keywords_clean"] = df["keywords"].fillna("").apply(extract_names)
        df["overview"] = df["overview"].fillna("")
        df["year"] = pd.to_datetime(
            df["release_date"], errors="coerce"
        ).dt.year.astype("Int64")

        df["content"] = (
            df["genres_clean"] + " " +
            df["keywords_clean"] + " " +
            df["overview"]
        ).str.strip()

        df = df[df["title"].notna()].copy()
        df["content"] = df["content"].fillna("")

        return df

    @st.cache_data
    def compute_similarity(df):
        tfidf = TfidfVectorizer(
        stop_words="english"
    )

        tfidf_matrix = tfidf.fit_transform(df["content"])

        return cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    df = load_data()
    cosine_sim = compute_similarity(df)

    indices = pd.Series(
        df.index,
        index=df["title"].str.strip().str.lower()
    ).drop_duplicates()

    def find_movie_title(title):
        query = str(title).strip().lower()

        if not query:
            return None

        if query in indices.index:
            return df.loc[indices[query], "title"]

        partial_matches = [
            original_title
            for original_title in df["title"].dropna().unique()
            if query in str(original_title).lower()
        ]

        if partial_matches:
            return partial_matches[0]

        title_map = {
            str(t).lower(): str(t)
            for t in df["title"].dropna().unique()
        }

        close = difflib.get_close_matches(
            query, list(title_map.keys()), n=1, cutoff=0.65
        )

        if close:
            return title_map[close[0]]

        return None

    def get_input_features(title):
        matched_title = find_movie_title(title)
        if matched_title is None:
            return None

        idx = indices[matched_title.lower()]
        movie = df.loc[idx]

        keywords_display = movie["keywords_clean"]
        if len(keywords_display) > 60:
            keywords_display = keywords_display[:60] + "..."

        return pd.DataFrame({
            "title": [movie["title"]],
            "year": [movie["year"]],
            "genres": [movie["genres_clean"].replace("_", " ")],
            "keywords": [keywords_display.replace("_", " ")]
        })

    def recommend(title, top_n=10):
        matched_title = find_movie_title(title)

        if matched_title is None:
            return None

        idx = indices[matched_title.lower()]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )[1:top_n + 1]

        movie_indices = [i for i, _ in sim_scores]
        scores = [round(score, 4) for _, score in sim_scores]

        result = df[
            ["title", "year", "genres_clean", "keywords_clean"]
        ].iloc[movie_indices].copy()

        result.insert(0, "rank", range(1, len(result) + 1))
        result["similarity_score"] = scores

        result["keywords_clean"] = result["keywords_clean"].apply(
            lambda x: ", ".join(x.split()[:5]) if x else ""
        )

        result = result.rename(columns={
            "title": "movie_title",
            "genres_clean": "genres",
            "keywords_clean": "keywords"
        })

        # Make the display easier to read without changing the features
        # used by the recommendation algorithm.
        result["genres"] = result["genres"].str.replace("_", " ", regex=False)
        result["keywords"] = result["keywords"].str.replace("_", " ", regex=False)

        return result.reset_index(drop=True)

    # ---------- UI ----------
    movie_list = sorted(df["title"].dropna().unique())

    input_method = st.radio(
        "Select input method:",
        ["Choose from list", "Type movie name"],
        horizontal=True,
        key="cb_input_method"
    )

    if input_method == "Choose from list":
        selected_movie = st.selectbox(
            "Choose a movie:", movie_list, key="cb_selectbox"
        )
    else:
        user_input = st.text_input(
            "Type a movie name:",
            key="cb_text_input"
        )
        selected_movie = None

        if user_input:
            selected_movie = find_movie_title(user_input)
            if selected_movie:
                st.success(f"✓ Found: {selected_movie}")
            else:
                st.warning(
                    "No matching movie found. Please try a different name."
                )

    top_n = st.slider(
        "Number of recommendations:",
        5, 20, 10,
        key="cb_slider"
    )

    if st.button("Get Recommendations", key="cb_button") and selected_movie:
        st.subheader("Input Movie Features")
        st.table(get_input_features(selected_movie))

        st.subheader("Recommended Movies")
        results = recommend(selected_movie, top_n)
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        avg_score = results["similarity_score"].mean()
        st.caption(
            f"Average similarity score of top {top_n} recommendations: "
            f"{avg_score:.4f}"
        )

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