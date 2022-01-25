import gc

import pandas as pd
import requests
import streamlit as st


@st.cache(show_spinner=False)
def import_data():
    """
    Import necessary data.
    Returns
    -------
    movie_: pandas.DataFrame
        Dataframe contains data from movie.pkl
    rating_: pandas.DataFrame
        Dataframe contains data from rating.pkl
    user_movie_df_: pandas.DataFrame
        Dataframe contains data from user_movie_df.pkl
    link_: pandas.DataFrame
        Dataframe contains data from link.pkl
    """
    gc.disable()
    link_ = pd.read_parquet("data/link.parquet")
    movie_ = pd.read_parquet("data/movie.parquet")
    rating_ = pd.read_parquet("data/rating.parquet")
    user_movie_df_ = pd.read_parquet("data/user_movie_df.parquet")
    gc.enable()
    return movie_, rating_, user_movie_df_, link_


def give_me_recommendations(movie_, rating_, user_movie_df_, id_: int):
    """
    Returns movie IDs based on User-based and Item-based recommendation.
    Parameters
    ----------
    movie_: pandas.DataFrame
    rating_: pandas.DataFrame
    user_movie_df_: pandas.DataFrame
    id_: int

    Returns
    -------
    user_5: pandas.Series
        Series contains most correlated 5 movie ids based on User-based
        recommendation.
    item_5: pandas.Series
        Series contains most correlated 5 movie ids based on Item-based
        recommendation.
    """
    user_df = user_movie_df_[user_movie_df_.index == id_]
    watched = user_df.columns[user_df.notna().any()].to_list()

    ###########################################################################

    movies_watched = user_movie_df_[watched]

    user_movie_count = movies_watched.T.notnull().sum().reset_index()
    user_movie_count.columns = ["userID", "movie_count"]
    user_movie_count = user_movie_count[user_movie_count.movie_count > 0]

    user_movie_count["similarity"] = pd.cut(
        user_movie_count["movie_count"], 100, labels=list(range(1, 101))
    )
    user_movie_count = user_movie_count.astype("int64")

    watched_same_movies = user_movie_count[
        user_movie_count["similarity"] >= 50].userID

    ###########################################################################

    final_df = pd.concat(
        [
            movies_watched[movies_watched.index.isin(watched_same_movies)],
            user_df[watched],
        ]
    )

    corr_df = final_df.T.corr().unstack().sort_values(
        ascending=False).drop_duplicates()

    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ["user_id_1", "user_id_2"]
    corr_df = corr_df.reset_index()

    top_users = \
    corr_df[(corr_df["user_id_1"] == id_) & (corr_df["corr"] >= 0.50)][
        ["user_id_2", "corr"]
    ].reset_index(drop=True)

    top_users_ratings = top_users.merge(
        rating_[["userId", "movieId", "rating"]],
        how="inner",
        left_on="user_id_2",
        right_on="userId",
    )

    top_users_ratings.drop("userId", axis=1, inplace=True)
    top_users_ratings = top_users_ratings[top_users_ratings.user_id_2 != id_]

    ###########################################################################

    top_users_ratings["weighted_rating"] = (
            top_users_ratings["corr"] * top_users_ratings["rating"]
    )

    recommendation = top_users_ratings.groupby(
        "movieId").weighted_rating.mean()
    recommendation = recommendation.reset_index().sort_values(
        "weighted_rating", ascending=False
    )

    user_5_ = movie_[movie_.movieId.isin(recommendation.movieId.iloc[:5])][
        "title"]

    ###########################################################################
    try:
        movie_id = (
            rating_[
                (rating_["userId"] == id_) & (
                            rating_["rating"] == rating_["rating"].max())
                ]
                .sort_values("timestamp", ascending=False)["movieId"][:1]
                .values[0]
        )


        movie_name = user_movie_df_[
            movie_[movie_.movieId == movie_id].title.values[0]]

        item_5_ = (
            user_movie_df_.corrwith(movie_name)
                .sort_values(ascending=False)
                .iloc[:6]
                .iloc[1:]
        )
    except (KeyError, IndexError):
        movie_id = (
            rating_[
                (rating_["userId"] == id_) & (
                        rating_["rating"] == rating_["rating"].max())
                ]
                .sort_values("timestamp", ascending=False)["movieId"][1:2]
                .values[0]
        )
        movie_id = (
            rating_[
                (rating_["userId"] == id_)
                & (rating_["rating"] == rating_["rating"].max())
                ]
                .sort_values("timestamp", ascending=False)["movieId"][1:2]
                .values[0]
        )
        movie_name = user_movie_df_[
            movie_[movie_.movieId == movie_id].title.values[0]]

        item_5_ = (
            user_movie_df_.corrwith(movie_name)
                .sort_values(ascending=False)
                .iloc[:6]
                .iloc[1:]
        )

    return user_5_, item_5_


def id_generator(top5user, top5item, movie_df_, link_df_):
    """
    Generate TMDB ID's and titles.
    Parameters
    ----------
    top5user: pd.Series
        Contains most correlated 5 movie ids based on User-based recommendation.
    top5item: pd.Series
        Contains most correlated 5 movie ids based on Item-based recommendation.
    movie_df_: pd.DataFrame
        Dataframe contains data from movie.pkl
    link_df_:
        Dataframe contains data from link.pkl
    Returns
    -------
    user_based.tmdbId.values: list
        TMDB ID's from User-based recommendation.
    item_based.tmdbId.values: list
        TMDB ID's paths from Item-based recommendation.
    user_based.title.values: list
        Movie titles from User-based recommendation.
    item_based.title.values: list
        Movie titles from Item-based recommendation.
    """
    user_based = (
        top5user.to_frame()
            .merge(movie_df_, on="title", how="left")
            .merge(link_df_, on="movieId", how="left")[["title", "tmdbId"]]
    )
    item_based = (
        top5item.to_frame()
            .merge(movie_df_, on="title", how="left")
            .merge(link_df_, on="movieId", how="left")[["title", "tmdbId"]]
    )
    try:
        user_based.tmdbId = user_based.tmdbId.astype("int64")
    except pd.errors.IntCastingNaNError:
        pass
    try:
        item_based.tmdbId = item_based.tmdbId.astype("int64")
    except pd.errors.IntCastingNaNError:
        pass

    return (
        user_based.tmdbId.values,
        item_based.tmdbId.values,
        user_based.title.values,
        item_based.title.values,
    )


def posters(id_list_):
    """
    Generates poster paths for TMDB API.
    Parameters
    ----------
    id_list_: list
        TMDB ID's from recommendations.
    Returns
    -------
    paths: list
        List containing poster paths.
    """
    api_key = st.secrets["api_key"]
    paths = []
    for _ in id_list_:
        try:
            query = f"https://api.themoviedb.org/3/movie/{_}?api_key={api_key}"
            response = requests.get(query)
            mov = response.json()
            paths.append(mov["poster_path"])
        except KeyError:
            pass
    return paths
