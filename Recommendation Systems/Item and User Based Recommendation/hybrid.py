#################################### PREP #####################################

import pandas as pd

pd.set_option('display.max_columns', None)

###############################################################################

movie = pd.read_csv('movie_lens_dataset/movie.csv')
rating = pd.read_csv('movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.shape  # 20000797, 6
print(*list(df.columns), sep='\n')
df.isnull().sum()
df.dropna(inplace=True)
len(df[(df.rating < 0) | (df.rating > 5)])  # 0
df.userId = df.userId.astype('int64')
df.shape
df.head()

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
user_movie_df = common_movies.pivot_table(index=["userId"],
                                          columns=["title"],
                                          values="rating")

user_movie_df.shape  # 138493 users, 3159 movies
user_movie_df.head()

###############################################################################
# User Selection
random_user = int(pd.Series(user_movie_df.index). \
                  sample(1, random_state=101). \
                  values)  # 2854
user_df = user_movie_df[user_movie_df.index == random_user]
watched = user_df.columns[user_df.notna().any()].to_list()
len(watched)  # 54 movies
print(*watched, sep='\n')

###############################################################################

# Ana matristen kullanıcının izlediği filmlerin filtrelenmesi
movies_watched = user_movie_df[watched]
movies_watched.shape  # 138493 users, 54 movies
movies_watched.T.head()

# Ortak izlenmiş ve oylanmış film sayısı
user_movie_count = movies_watched.T.notnull().sum().reset_index()
user_movie_count.columns = ['userID', 'movie_count']
user_movie_count = user_movie_count[user_movie_count.movie_count > 0]
user_movie_count.movie_count.describe()
user_movie_count.head()

# Belirli alışkanlıkların benzerliği belirlenmeye çalışıldığı için ortak izlenmiş
# film sayıları düşük olan kullanıcıları filtrelebiliriz. Kullanıcılar arasındaki
# ortak izlenmiş filmlerin oranı %60'tan düşük kullanıcılar filtreleyelim.

user_movie_count['similarity'] = pd.cut(user_movie_count['movie_count'],
                                        100,
                                        labels=[i for i in range(1, 101)])
user_movie_count = user_movie_count.astype('int64')

watched_same_movies = user_movie_count[
    user_movie_count["similarity"] > 60].userID
watched_same_movies.count()  # 245 kullanıcı
list(watched_same_movies)  # userID list

similar_user_data = movies_watched[
    (movies_watched.index.isin(watched_same_movies)) &
    (movies_watched.index != random_user)]
similar_user_data.head()  # Benzer kullanıcıların verileri

###############################################################################

final_df = pd.concat(
    [movies_watched[movies_watched.index.isin(watched_same_movies)],
     user_df[watched]])

corr_df = final_df.T.corr().unstack().sort_values(ascending=False). \
    drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=['corr'])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()

top_users = corr_df[(corr_df['user_id_1'] == random_user) &
                    (corr_df['corr'] >= 0.60)][['user_id_2', 'corr']]. \
    reset_index(drop=True)
top_users.head()
top_users_ratings = top_users.merge(rating[['userId', 'movieId', 'rating']],
                                    how='inner', left_on='user_id_2',
                                    right_on='userId')

top_users_ratings.drop('userId', axis=1, inplace=True)
top_users_ratings = top_users_ratings[
    top_users_ratings.user_id_2 != random_user]

###############################################################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * \
                                       top_users_ratings['rating']

recommendation = top_users_ratings.groupby('movieId').weighted_rating.mean()
recommendation = recommendation.reset_index().sort_values('weighted_rating',
                                                          ascending=False)

recommendation.head()
user_5 = movie[movie.movieId.isin(recommendation.movieId.iloc[:5])]['title']

###############################################################################

# Item Based Önerinin Hazırlanması
movie_id = rating[(rating['userId'] == random_user) & (rating['rating'] == 5)]. \
               sort_values('timestamp', ascending=False)['movieId'][
           0:1].values[0]

movie_name = user_movie_df[movie[movie.movieId == movie_id].title.values[0]]

item_5 = user_movie_df.corrwith(movie_name).sort_values(ascending=False). \
             head(6).iloc[1:]

recos = pd.DataFrame({'USER BASED': user_5.values, 'ITEM BASED': item_5.index},
                     index=[i for i in range(1, 6)])

# Sonuçlar
print(recos)
# ya da
user = "\n".join(list(user_5))
item = "\n".join(list(item_5.index))
print(f"USER BASED:\n{user}", sep='\n')
print('------------------------------')
print(f"ITEM BASED:\n{item}", sep='\n')

# USER BASED:
# Underground (1995)
# Murder, My Sweet (1944)
# Mirror, The (Zerkalo) (1975)
# Come and See (Idi i smotri) (1985)
# Samouraï, Le (Godson, The) (1967)
# ------------------------------
# ITEM BASED:
# Monty Python and the Holy Grail (1975)
# Monty Python's The Meaning of Life (1983)
# Monty Python Live at the Hollywood Bowl (1982)
# Monty Python's And Now for Something Completely Different (1971)
# Battle of Algiers, The (La battaglia di Algeri) (1966)
