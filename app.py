import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast
import difflib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
}

p, label {
    color: #111827;
}

/* TITLE */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #6b7280 !important;
    margin-bottom: 30px;
}


/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #111827;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-title {
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    font-size: 14px;
    color: #d1d5db !important;
    margin-bottom: 25px;
}


/* CARDS */

.feature-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.feature-title {
    font-size: 20px;
    font-weight: 700;
    color: #111827;
}

.feature-description {
    color: #6b7280 !important;
    font-size: 14px;
    margin-top: 5px;
}


/* SECTION */

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 15px;
}


/* BUTTON */

.stButton > button {
    width: 100%;
    border-radius: 9px;
    font-weight: 700;
    padding: 0.6rem 1rem;
}


/* INFO */

.info-box {
    background: #eef2ff;
    border-left: 5px solid #4f46e5;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.info-box-title {
    font-weight: 700;
    color: #3730a3;
}

.info-box-text {
    color: #4b5563 !important;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎬 Movie Recommender System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover movies using Content-Based, Collaborative, and Hybrid Filtering'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD CONTENT DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/tmdb_movie_dataset.csv"
    )

    def extract_names(text):

        try:

            items = ast.literal_eval(text)

            if isinstance(items, list):

                return " ".join(
                    str(item.get("name", ""))
                    for item in items
                    if isinstance(item, dict)
                )

        except (
            ValueError,
            SyntaxError,
            TypeError
        ):
            pass

        return ""

    df["genres_clean"] = (
        df["genres"]
        .fillna("")
        .apply(extract_names)
    )

    df["keywords_clean"] = (
        df["keywords"]
        .fillna("")
        .apply(extract_names)
    )

    df = df[
        df["title"].notna()
    ].copy()

    return df


@st.cache_data
def compute_similarity(df):
    """
    Calculate cosine similarity using movie genres and keywords.
    """
    
    all_features = (
        df["genres_clean"].fillna("") + " " +
        df["keywords_clean"].fillna("")
    ).str.strip()

    # Create binary feature vectors
    feature_list = sorted(
        set(
            " ".join(all_features).split()
        )
    )

    feature_index = {
        feature: i
        for i, feature in enumerate(feature_list)
    }

    matrix = []

    for content in all_features:
        vector = [0] * len(feature_list)

        for feature in content.split():
            if feature in feature_index:
                vector[feature_index[feature]] = 1

        matrix.append(vector)

    return cosine_similarity(matrix, matrix)


df = load_data()

cosine_sim = compute_similarity(df)


# ============================================================
# MOVIE INDEX
# ============================================================

indices = pd.Series(
    df.index,
    index=df["title"]
    .astype(str)
    .str.strip()
    .str.lower()
).drop_duplicates()


# ============================================================
# FIND MOVIE
# ============================================================

def find_movie_title(title):

    query = str(title).strip().lower()

    if not query:
        return None

    # Exact match
    if query in indices.index:

        return df.loc[
            indices[query],
            "title"
        ]

    # Partial match
    partial_matches = [

        original_title

        for original_title
        in df["title"]
        .dropna()
        .unique()

        if query in str(
            original_title
        ).lower()
    ]

    if partial_matches:

        return partial_matches[0]

    # Fuzzy match
    title_map = {

        str(t).strip().lower():
            str(t)

        for t
        in df["title"]
        .dropna()
        .unique()
    }

    close = difflib.get_close_matches(
        query,
        list(title_map.keys()),
        n=1,
        cutoff=0.65
    )

    if close:

        return title_map[
            close[0]
        ]

    return None


# ============================================================
# CONTENT INPUT FEATURES
# ============================================================

def get_input_features(title):

    matched_title = find_movie_title(
        title
    )

    if matched_title is None:

        return None

    idx = indices[
        matched_title.lower()
    ]

    movie = df.loc[idx]

    keywords_display = (
        ", ".join(
            dict.fromkeys(
                movie["keywords_clean"].split()
            )
        )
        if movie["keywords_clean"]
        else ""
    )

    return pd.DataFrame({

        "Movie": [
            movie["title"]
        ],

        "Genres": [
            movie["genres_clean"]
            .replace("_", " ")
        ],

        "Keywords": [
            keywords_display
            .replace("_", " ")
        ]
    })


# ============================================================
# CONTENT RECOMMENDATION
# ============================================================

def recommend(title, top_n=10):

    matched_title = find_movie_title(title)

    if matched_title is None:
        return None

    idx = indices[
        matched_title.lower()
    ]

    input_movie = df.loc[idx]

    # -----------------------------
    # Input movie genres
    # -----------------------------

    input_genres = set(
        str(input_movie["genres_clean"]).split()
    )

    # -----------------------------
    # Input movie keywords
    # -----------------------------

    input_keywords = set(
        str(input_movie["keywords_clean"]).split()
    )

    candidates = []

    for i, movie in df.iterrows():

        # Do not recommend the input movie itself
        if i == idx:
            continue

        movie_genres = set(
            str(movie["genres_clean"]).split()
        )

        movie_keywords = set(
            str(movie["keywords_clean"]).split()
        )

        # -----------------------------
        # Genre Match
        # -----------------------------

        common_genres = (
            input_genres & movie_genres
        )

        if len(input_genres) == 0:
            genre_match = 0
        else:
            genre_match = (
                len(common_genres)
                / len(input_genres)
            )

        # At least 50% genre match
        if genre_match < 0.50:
            continue

        # -----------------------------
        # Keyword Match
        # -----------------------------

        common_keywords = (
            input_keywords & movie_keywords
        )

        keyword_match_count = len(
            common_keywords
        )

        # At least 2 common keywords
        if keyword_match_count < 2:
            continue

        # -----------------------------
        # Cosine Similarity
        # -----------------------------

        similarity_score = cosine_sim[
            idx
        ][i]

        candidates.append({
            "index": i,
            "genre_match": genre_match,
            "keyword_matches": keyword_match_count,
            "similarity_score": similarity_score
        })

    # -----------------------------
    # Sort by Cosine Similarity
    # -----------------------------

    candidates = sorted(
        candidates,
        key=lambda x: x["similarity_score"],
        reverse=True
    )

    candidates = candidates[:top_n]

    if not candidates:
        return pd.DataFrame(
            columns=[
                "Rank",
                "Movie Title",
                "Genres",
                "Keywords",
                "Genre Match",
                "Keyword Matches",
                "Similarity Score"
            ]
        )

    movie_indices = [
        item["index"]
        for item in candidates
    ]

    result = df.loc[
        movie_indices,
        [
            "title",
            "genres_clean",
            "keywords_clean"
        ]
    ].copy()

    result.insert(
        0,
        "Rank",
        range(
            1,
            len(result) + 1
        )
    )

    result["Genre Match"] = [
        round(
            item["genre_match"] * 100,
            1
        )
        for item in candidates
    ]

    result["Keyword Matches"] = [
        item["keyword_matches"]
        for item in candidates
    ]

    result["Similarity Score"] = [
        round(
            item["similarity_score"],
            4
        )
        for item in candidates
    ]

    result["keywords_clean"] = (
        result["keywords_clean"]
        .apply(
            lambda x:
            ", ".join(
                dict.fromkeys(x.split())
            )
            if x else ""
        )
    )

    result = result.rename(
        columns={
            "title": "Movie Title",
            "genres_clean": "Genres",
            "keywords_clean": "Keywords"
        }
    )

    result["Genres"] = (
        result["Genres"]
        .str.replace(
            "_",
            " ",
            regex=False
        )
    )

    result["Keywords"] = (
        result["Keywords"]
        .str.replace(
            "_",
            " ",
            regex=False
        )
    )

    return result.reset_index(
        drop=True
    )

# ============================================================
# LOAD COLLABORATIVE DATA
# ============================================================

@st.cache_data
def load_collaborative_data():

    ratings = pd.read_csv(
        "data/tmdb_movie_ratings.csv",
        usecols=[
            "userId",
            "ratingId",
            "rating"
        ]
    )

    ratings = ratings[
        ratings["userId"] <= 600
    ].copy()

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


# ============================================================
# MOVIE STATISTICS
# ============================================================

movie_stats = (
    data_cf
    .groupby("title")["rating"]
    .agg(
        average_rating="mean",
        number_of_ratings="count"
    )
    .reset_index()
)

movie_stats[
    "average_rating"
] = (
    movie_stats[
        "average_rating"
    ]
    .round(2)
)


# ============================================================
# USER MOVIE MATRIX
# ============================================================

movie_matrix = data_cf.pivot_table(
    index="userId",
    columns="title",
    values="rating"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🎬 Movie AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">'
        'Intelligent Movie Recommendation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        "### 📊 System Overview"
    )

    st.write(
        f"🎞️ Movies: **{len(df):,}**"
    )

    st.write(
        f"👥 Users: **{data_cf['userId'].nunique():,}**"
    )

    st.write(
        f"⭐ Ratings: **{len(data_cf):,}**"
    )

    st.markdown("---")

    st.markdown(
        "### 🤖 Algorithms"
    )

    st.write(
        "🎯 Content-Based"
    )

    st.write(
        "👥 Collaborative"
    )

    st.write(
        "🔀 Hybrid Filtering"
    )

    st.markdown("---")

    st.caption(
        "Movie Recommender System\n"
        "Group Project"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Content-Based",
        "👥 Collaborative",
        "🔀 Hybrid"
    ]
)


# ============================================================
# TAB 1 - CONTENT BASED
# ============================================================

with tab1:

    st.header(
        "🎯 Content-Based Filtering"
    )

    st.write(
        "Find movies with similar genres and keywords."
    )

    st.markdown(
        '<div class="feature-card">'
        '<div class="feature-title">'
        'Genre + Keyword + Cosine Similarity'
        '</div>'
        '<div class="feature-description">'
        'The system first filters movies by genre and keyword relevance, then uses cosine similarity to rank the final recommendations.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    movie_list = sorted(
        df["title"]
        .dropna()
        .unique()
    )

    default_movie = (
        df.sort_values("vote_count", ascending=False)
        .iloc[0]["title"]
    )

    default_index = (
        movie_list.index(default_movie)
        if default_movie in movie_list
        else 0
    )

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        selected_movie = st.selectbox(
            "Choose a movie:",
            movie_list,
            index=default_index,
            key="cb_selectbox"
        )

    with col2:

        top_n = st.slider(
            "Recommendations",
            5,
            20,
            10,
            key="cb_slider"
        )

    if st.button(
        "🎬 Get Recommendations",
        key="cb_button",
        use_container_width=True
    ):

        features = get_input_features(
            selected_movie
        )

        st.markdown(
            '<div class="section-title">'
            '🎞️ Selected Movie'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            features,
            use_container_width=True,
            hide_index=True
        )

        results = recommend(
            selected_movie,
            top_n
        )

        st.markdown(
            '<div class="section-title">'
            '🍿 Recommended Movies'
            '</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        # =================================================
        # WHY THIS MOVIE?
        # =================================================

        def generate_cb_reason(row):

            genre_match = row["Genre Match"]
            keyword_matches = row["Keyword Matches"]
            similarity = row["Similarity Score"]

            if genre_match >= 80 and keyword_matches >= 5:
                return (
                    "Very similar genres and many "
                    "shared keywords with the selected movie"
                )
            elif genre_match >= 80:
                return (
                    "Strong genre similarity "
                    "with the selected movie"
                )
            elif keyword_matches >= 5:
                return (
                    "Shares many keywords/themes "
                    "with the selected movie"
                )
            elif similarity >= 0.50:
                return (
                    "High overall content similarity "
                    "with the selected movie"
                )
            else:
                return (
                    "Meets the minimum genre and "
                    "keyword overlap requirements"
                )

        if not results.empty:

            st.markdown(
                "### 💡 Why this movie?"
            )

            st.caption(
                "The explanation is based on genre "
                "overlap, keyword matches, and "
                "cosine similarity."
            )

            for _, row in results.iterrows():

                reason = generate_cb_reason(row)

                st.markdown(
                    f"**{int(row['Rank'])}. "
                    f"{row['Movie Title']}**"
                )

                st.info(
                    f"💡 {reason}"
                )

        # =================================================
        # METRICS
        # =================================================

        if not results.empty:

            avg_score = (
                results[
                    "Similarity Score"
                ].mean()
            )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Recommendations",
                    len(results)
                )

            with c2:

                st.metric(
                    "Average Similarity",
                    f"{avg_score:.4f}"
                )
        else:

            st.warning(
                "No recommendations found for this movie."
            )

# ============================================================
# TAB 2 - COLLABORATIVE
# ============================================================

with tab2:

    st.header(
        "👥 Collaborative Filtering"
    )

    st.write(
        "Recommend movies based on "
        "user rating behaviour."
    )

    st.markdown(
        '<div class="feature-card">'
        '<div class="feature-title">'
        'User Rating Behaviour'
        '</div>'
        '<div class="feature-description">'
        'Movies are recommended based on '
        'similar rating patterns from users.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    movie_list_cf = sorted(
        movie_matrix.columns.tolist()
    )
    default_movie_cf = (
        df.sort_values("vote_count", ascending=False)
        .iloc[0]["title"]
    )

    default_index_cf = (
        movie_list_cf.index(default_movie_cf)
        if default_movie_cf in movie_list_cf
        else 0
    )
    

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        selected_movie_cf = st.selectbox(
            "Choose a movie:",
            movie_list_cf,
            index=default_index_cf,
            key="cf_movie"
        )

    with col2:

        top_n_cf = st.slider(
            "Recommendations",
            5,
            20,
            10,
            key="cf_slider"
        )

    if st.button(
        "👥 Get Collaborative Recommendations",
        key="cf_button",
        use_container_width=True
    ):

        similar = movie_matrix.corrwith(
            movie_matrix[
                selected_movie_cf
            ]
        )

        corr = pd.DataFrame(
            similar,
            columns=[
                "Correlation"
            ]
        )

        corr.dropna(
            inplace=True
        )

        recommendations = (
            corr
            .drop(
                selected_movie_cf
            )
        )

        recommendations = (
            recommendations
            .join(
                movie_stats.set_index(
                    "title"
                )[
                    [
                        "number_of_ratings"
                    ]
                ]
            )
        )

        recommendations = (
            recommendations[
                recommendations[
                    "number_of_ratings"
                ] >= 100
            ]
        )

        recommendations = (
            recommendations
            .sort_values(
                "Correlation",
                ascending=False
            )
            .head(top_n_cf)
        )

        selected_info = data_cf[
            data_cf["title"]
            == selected_movie_cf
        ].iloc[0]

        selected_stats = movie_stats[
            movie_stats["title"]
            == selected_movie_cf
        ].iloc[0]

        st.markdown(
            '<div class="section-title">'
            '🎞️ Selected Movie'
            '</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Movie",
                selected_movie_cf
            )

        with c2:

            st.metric(
                "Average Rating",
                selected_stats[
                    "average_rating"
                ]
            )

        with c3:

            st.metric(
                "User Ratings",
                int(
                    selected_stats[
                        "number_of_ratings"
                    ]
                )
            )

        with c4:

            st.metric(
                "TMDB Rating",
                selected_info[
                    "vote_average"
                ]
            )

        st.markdown(
            '<div class="section-title">'
            '🍿 Recommended Movies'
            '</div>',
            unsafe_allow_html=True
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

            result.insert(
                0,
                "Rank",
                range(
                    1,
                    len(result) + 1
                )
            )

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

            result[
                "Correlation Score"
            ] = (
                result[
                    "Correlation Score"
                ].round(4)
            )

            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

            avg_correlation = (
                recommendations[
                    "Correlation"
                ].mean()
            )

            st.metric(
                "Average Correlation Score",
                f"{avg_correlation:.4f}"
            )


# ============================================================
# TAB 3 - HYBRID
# ============================================================
with tab3:

    st.header(
        "🔀 Hybrid Filtering"
    )

    st.write(
        "Combines Content-Based and "
        "Collaborative Filtering."
    )

    st.markdown(
        '<div class="info-box">'
        '<div class="info-box-title">'
        '💡 Hybrid Recommendation'
        '</div>'
        '<div class="info-box-text">'
        'The hybrid system combines movie content '
        'similarity with user rating behaviour '
        'to produce a more balanced recommendation.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # ONLY USE MOVIES THAT EXIST IN BOTH SYSTEMS
    # --------------------------------------------------------

    hybrid_movies = sorted(
        set(
            df["title"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        &
        set(
            movie_matrix.columns
            .astype(str)
            .str.strip()
        )
    )
    default_hybrid = (
            df.sort_values("vote_count", ascending=False)
            .iloc[0]["title"]
        )

    default_index_hybrid = (
            hybrid_movies.index(default_hybrid)
            if default_hybrid in hybrid_movies
            else 0
        )

    if not hybrid_movies:

        st.error(
            "No common movies found between "
            "Content-Based and Collaborative datasets."
        )

    else:

        selected_hybrid_movie = st.selectbox(
            "Choose a movie:",
            hybrid_movies,
            index=default_index_hybrid,
            key="hybrid_movie"
        )

        st.markdown(
            "### ⚖️ Recommendation Weights"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            content_weight = st.slider(
                "🎯 Content Weight",
                0.20,
                0.80,
                0.50,
                0.05,
                key="hybrid_content_weight_v2"
            )

        with col2:

            collaborative_weight = (
                1.0
                - content_weight
            )

            st.metric(
                "👥 Collaborative Weight",
                f"{collaborative_weight:.0%}"
            )

        with col3:

            hybrid_top_n = st.slider(
                "Recommendations",
                5,
                20,
                10,
                key="hybrid_slider"
            )

        st.caption(
            f"Hybrid Formula: "
            f"(Content × {content_weight:.0%}) + "
            f"(Collaborative × "
            f"{collaborative_weight:.0%})"
        )

        if st.button(
            "🔀 Generate Hybrid Recommendations",
            key="hybrid_button",
            use_container_width=True
        ):

            # =================================================
            # CONTENT SCORE
            # =================================================

            content_title = find_movie_title(
                selected_hybrid_movie
            )

            if content_title is None:

                st.error(
                    "Selected movie could not "
                    "be found in the content dataset."
                )

                st.stop()

            content_idx = indices[
                content_title.lower()
            ]

            content_score_df = pd.DataFrame({
                "Movie Title":
                    df["title"]
                    .astype(str)
                    .str.strip(),

                "Content Score":
                    cosine_sim[
                        content_idx
                    ]
            })

            content_score_df = (
                content_score_df
                .drop_duplicates(
                    subset="Movie Title",
                    keep="first"
                )
            )

            # =================================================
            # COLLABORATIVE SCORE
            # =================================================

            similar = movie_matrix.corrwith(
                movie_matrix[
                    selected_hybrid_movie
                ]
            )

            collaborative_score_df = (
                similar
                .dropna()
                .reset_index()
            )

            collaborative_score_df.columns = [
                "Movie Title",
                "Collaborative Score"
            ]

            collaborative_score_df[
                "Movie Title"
            ] = (
                collaborative_score_df[
                    "Movie Title"
                ]
                .astype(str)
                .str.strip()
            )

            collaborative_score_df = (
                collaborative_score_df
                .drop_duplicates(
                    subset="Movie Title",
                    keep="first"
                )
            )

            # =================================================
            # MERGE BOTH SCORES
            # =================================================

            hybrid_df = content_score_df.merge(
                collaborative_score_df,
                on="Movie Title",
                how="inner"
            )

            # Remove selected movie

            hybrid_df = hybrid_df[
                hybrid_df[
                    "Movie Title"
                ]
                != selected_hybrid_movie
            ].copy()

            # =================================================
            # REMOVE INVALID VALUES
            # =================================================

            hybrid_df = hybrid_df.dropna(
                subset=[
                    "Content Score",
                    "Collaborative Score"
                ]
            )

            # =================================================
            # NORMALISE SCORES
            # =================================================

            def min_max(series):

                minimum = series.min()
                maximum = series.max()

                if (
                    pd.isna(minimum)
                    or pd.isna(maximum)
                    or maximum == minimum
                ):

                    return pd.Series(
                        0.0,
                        index=series.index
                    )

                return (
                    (series - minimum)
                    /
                    (maximum - minimum)
                )

            hybrid_df[
                "Content Score"
            ] = min_max(
                hybrid_df[
                    "Content Score"
                ]
            )

            hybrid_df[
                "Collaborative Score"
            ] = min_max(
                hybrid_df[
                    "Collaborative Score"
                ]
            )

            # =================================================
            # HYBRID SCORE
            # =================================================

            hybrid_df[
                "Hybrid Score"
            ] = (
                hybrid_df[
                    "Content Score"
                ]
                * content_weight
                +
                hybrid_df[
                    "Collaborative Score"
                ]
                * collaborative_weight
            )

            # =================================================
            # SORT
            # =================================================

            hybrid_df = (
                hybrid_df
                .sort_values(
                    "Hybrid Score",
                    ascending=False
                )
                .head(
                    hybrid_top_n
                )
                .reset_index(
                    drop=True
                )
            )

            # =================================================
            # RANK
            # =================================================

            hybrid_df.insert(
                0,
                "Rank",
                range(
                    1,
                    len(hybrid_df) + 1
                )
            )

            # =================================================
            # WHY THIS MOVIE?
            # =================================================

            def generate_reason(row):

                cf = row[
                    "Collaborative Score"
                ]

                content = row[
                    "Content Score"
                ]

                if (
                    cf >= 0.80
                    and
                    content >= 0.80
                ):

                    return (
                        "Strong user-rating similarity "
                        "and strong content similarity"
                    )

                elif cf >= 0.80:

                    return (
                        "Users with similar rating "
                        "behaviour also liked this movie"
                    )

                elif content >= 0.80:

                    return (
                        "Strong similarity in genres, "
                        "keywords and movie content"
                    )

                elif (
                    cf >= 0.60
                    and
                    content >= 0.60
                ):

                    return (
                        "Good balance of user-rating "
                        "and content similarity"
                    )

                elif cf >= content:

                    return (
                        "Recommended mainly from "
                        "similar user-rating behaviour"
                    )

                else:

                    return (
                        "Recommended mainly from "
                        "similar movie content"
                    )

            hybrid_df[
                "Why this movie?"
            ] = hybrid_df.apply(
                generate_reason,
                axis=1
            )

            # =================================================
            # ROUND SCORES
            # =================================================

            hybrid_df[
                "Content Score"
            ] = (
                hybrid_df[
                    "Content Score"
                ].round(4)
            )

            hybrid_df[
                "Collaborative Score"
            ] = (
                hybrid_df[
                    "Collaborative Score"
                ].round(4)
            )

            hybrid_df[
                "Hybrid Score"
            ] = (
                hybrid_df[
                    "Hybrid Score"
                ].round(4)
            )

            # =================================================
            # DISPLAY
            # =================================================

            if hybrid_df.empty:

                st.warning(
                    "No hybrid recommendations found."
                )

            else:

                st.markdown(
                    '<div class="section-title">'
                    '🍿 Hybrid Recommendations'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.dataframe(
                    hybrid_df,
                    use_container_width=True,
                    hide_index=True
                )

                # =================================================
                # WHY THIS MOVIE? SECTION
                # =================================================

                st.markdown(
                    "### 💡 Why this movie?"
                )

                st.caption(
                    "The explanation is based on the "
                    "Content-Based and Collaborative "
                    "Filtering scores."
                )

                for _, row in hybrid_df.iterrows():

                    st.markdown(
                        f"**{int(row['Rank'])}. "
                        f"{row['Movie Title']}**"
                    )

                    st.info(
                        f"💡 {row['Why this movie?']}"
                    )

                # =================================================
                # METRICS
                # =================================================

                avg_hybrid = (
                    hybrid_df[
                        "Hybrid Score"
                    ].mean()
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "🎯 Content Weight",
                        f"{content_weight:.0%}"
                    )

                with c2:

                    st.metric(
                        "👥 Collaborative Weight",
                        f"{collaborative_weight:.0%}"
                    )

                with c3:

                    st.metric(
                        "🔀 Average Hybrid Score",
                        f"{avg_hybrid:.4f}"
                    )

                st.success(
                    "Hybrid recommendations generated successfully!"
                )
