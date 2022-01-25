import streamlit as st

from functions import import_data, give_me_recommendations, id_generator, \
    posters

st.markdown(
    "<h2 style='text-align: center; color: white;'>Hybrid Movie Recommender</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: red;'>Working process is not optimized.</p>",
    unsafe_allow_html=True,
)
selected_id = st.number_input("User ID: ", 1, 138493)
with st.spinner(text="Importing..."):
    movie_df, rating_df, user_movie_df, link_df = import_data()

with st.spinner(text="Working..."):
    user_5, item_5 = give_me_recommendations(
        movie_df, rating_df, user_movie_df, selected_id
    )

    user_based_ids, item_based_ids, user_based_titles, item_based_titles = id_generator(
        user_5, item_5, movie_df, link_df
    )

    user_reco_paths = posters(user_based_ids)
    item_reco_paths = posters(item_based_ids)

(
    col1,
    col2,
    col3,
    col4,
    col5,
) = st.columns(5)
col6, col7, col8, col9, col10 = st.columns(5)

try:
    col1.image(
        f"https://image.tmdb.org/t/p/w185/{user_reco_paths[0]}",
        caption=user_based_titles[0],
    )
    col2.image(
        f"https://image.tmdb.org/t/p/w185/{user_reco_paths[1]}",
        caption=user_based_titles[1],
    )
    col3.image(
        f"https://image.tmdb.org/t/p/w185/{user_reco_paths[2]}",
        caption=user_based_titles[2],
    )
    col4.image(
        f"https://image.tmdb.org/t/p/w185/{user_reco_paths[3]}",
        caption=user_based_titles[3],
    )
    col5.image(
        f"https://image.tmdb.org/t/p/w185/{user_reco_paths[4]}",
        caption=user_based_titles[4],
    )
except IndexError:
    pass
except KeyError:
    pass

try:
    col6.image(
        f"https://image.tmdb.org/t/p/w185/{item_reco_paths[0]}",
        caption=item_based_titles[0],
    )
    col7.image(
        f"https://image.tmdb.org/t/p/w185/{item_reco_paths[1]}",
        caption=item_based_titles[1],
    )
    col8.image(
        f"https://image.tmdb.org/t/p/w185/{item_reco_paths[2]}",
        caption=item_based_titles[2],
    )
    col9.image(
        f"https://image.tmdb.org/t/p/w185/{item_reco_paths[3]}",
        caption=item_based_titles[3],
    )
    col10.image(
        f"https://image.tmdb.org/t/p/w185/{item_reco_paths[4]}",
        caption=item_based_titles[4],
    )
except IndexError:
    pass
except KeyError:
    pass

user_reco_paths = []
item_reco_paths = []
