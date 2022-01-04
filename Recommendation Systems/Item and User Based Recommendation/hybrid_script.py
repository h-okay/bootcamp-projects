import pandas as pd
from cachetools import cached, TTLCache

cache = TTLCache(maxsize=100, ttl=86400)


@cached(cache)
def import_data():
    movie_ = pd.read_pickle('script_data/movie.pkl')
    rating_ = pd.read_pickle('script_data/rating.pkl')
    user_movie_df_ = pd.read_pickle('script_data/user_movie_df.pkl')
    return movie_, rating_, user_movie_df_


def give_me_recommendations(movie_, rating_, user_movie_df_, id_: int):
    user_df = user_movie_df_[user_movie_df_.index == id_]
    watched = user_df.columns[user_df.notna().any()].to_list()

    ###########################################################################

    movies_watched = user_movie_df_[watched]

    user_movie_count = movies_watched.T.notnull().sum().reset_index()
    user_movie_count.columns = ['userID', 'movie_count']
    user_movie_count = user_movie_count[user_movie_count.movie_count > 0]

    user_movie_count['similarity'] = pd.cut(user_movie_count['movie_count'],
                                            100,
                                            labels=[i for i in range(1, 101)])
    user_movie_count = user_movie_count.astype('int64')

    watched_same_movies = user_movie_count[
        user_movie_count["similarity"] > 60].userID

    ###########################################################################

    final_df = pd.concat(
        [movies_watched[movies_watched.index.isin(watched_same_movies)],
         user_df[watched]])

    corr_df = final_df.T.corr().unstack().sort_values(ascending=False). \
        drop_duplicates()

    corr_df = pd.DataFrame(corr_df, columns=['corr'])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df['user_id_1'] == id_) &
                        (corr_df['corr'] >= 0.60)][['user_id_2', 'corr']]. \
        reset_index(drop=True)

    top_users_ratings = top_users.merge(
        rating_[['userId', 'movieId', 'rating']],
        how='inner', left_on='user_id_2',
        right_on='userId')

    top_users_ratings.drop('userId', axis=1, inplace=True)
    top_users_ratings = top_users_ratings[
        top_users_ratings.user_id_2 != id_]

    ###########################################################################

    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * \
                                           top_users_ratings['rating']

    recommendation = top_users_ratings.groupby(
        'movieId').weighted_rating.mean()
    recommendation = recommendation.reset_index().sort_values(
        'weighted_rating',
        ascending=False)

    user_5 = movie_[movie_.movieId.isin(recommendation.movieId.iloc[:5])][
        'title']

    ###########################################################################

    # Item Based
    movie_id = \
        rating_[(rating_['userId'] == id_) & (rating_['rating'] == 5)]. \
            sort_values('timestamp', ascending=False)['movieId'][
        :1].values[0]

    movie_name = user_movie_df_[
        movie_[movie_.movieId == movie_id].title.values[0]]

    item_5 = user_movie_df_.corrwith(movie_name).sort_values(ascending=False). \
                 head(6).iloc[1:]

    user = "\n".join(list(user_5))
    item = "\n".join(list(item_5.index))
    if len(user) == 0:
        print(f"USER BASED:\nNone.", sep='\n')
        print('------------------------------')
        print(f"ITEM BASED:\n{item}", sep='\n')
    else:
        print(f"USER BASED:\n{user}", sep='\n')
        print('------------------------------')
        print(f"ITEM BASED:\n{item}", sep='\n')


if __name__ == '__main__':
    print('Booting up. Please wait...\nThis process can take up to 2 minutes.')
    movie, rating, user_movie_df = import_data()
    print('################# BOOT COMPLETE ####################')
    while True:
        while True:
            user_id = input('Please enter a user ID: ')
            if len(str(user_id)) > 5 or not user_id.isnumeric():
                continue
            break
        print('Working...\n')
        give_me_recommendations(movie, rating, user_movie_df, int(user_id))
        print('\n#####################################\n')
        while True:
            cont = input('Continue? [y/n] ').lower().strip()
            if cont == 'y':
                break
            elif cont == 'n':
                print('Goodbye.')
                break
            else:
                print('Invalid input!')
                continue
        if cont == 'y':
            continue
        else:
            break
