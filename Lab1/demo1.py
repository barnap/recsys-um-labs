import lenskit.datasets as ds
import pandas as pd
import csv
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser

data = ds.MovieLens('./dataset')
print("Successfully installed dataset.")
pd.set_option('display.max_columns', None)

rows_to_show = 10
print('####################################')
# print(data.ratings.head(rows_to_show))

joined_data = data.ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')

print(data.ratings.head(rows_to_show))
print(data.movies.head(rows_to_show))
print(data.movies['title'].head(rows_to_show))

pd.set_option('display.max_columns', None)
print(joined_data.head(rows_to_show))
print('####################################')

# RECOMMENDING MOVIES WITH THE HIGHEST AVERAGE RATE

average_ratings = data.ratings.groupby(['item']).mean()
sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[1:]]

print("RECOMMENDED MOVIES WITH THE HIGHEST AVERAGE RATE:")
print(joined_data.head(rows_to_show))

print('####################################')

# CHECK HOW MANY PEOPLE RATED THE MOVIES

average_ratings = data.ratings.groupby(['item'])\
    .agg(count=('user', 'size'), rating=('rating', 'mean'))\
    .reset_index()
sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
joined_data = joined_data[joined_data.columns[1:]]

print("CHECK HOW MANY PEOPLE RATED THE MOVIES:")
print(joined_data.head(rows_to_show))
print('####################################')

# RECOMMEND ONLY MOVIES RATED BY A MINIMUM NUMBER OF PEOPLE

minimum_to_include = 20
average_ratings = data.ratings.groupby(['item']).mean()
rating_counts = data.ratings.groupby(['item']).count()
average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
# joined_data = joined_data[joined_data.columns[3:]]

print("RECOMMEND ONLY MOVIES RATED BY A MINIMUM NUMBER OF PEOPLE:")
print(joined_data.head(rows_to_show))

print('####################################')

# FILTERING BY GENRE

genre = 'Action'

average_ratings = data.ratings.groupby(['item']).mean()
rating_counts = data.ratings.groupby(['item']).count()
average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
average_ratings = average_ratings.join(data.movies['genres'], on='item')
average_ratings = average_ratings.loc[average_ratings['genres'].str.contains(genre)]
sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_avg_ratings.join(data.movies['title'], on='item')
# joined_data = joined_data[joined_data.columns[3:]]
print("RECOMMENDED FOR THE GENRE ", genre)
print(joined_data.head(rows_to_show))

print('####################################')

jabril_rating_dict = {}

with open("C:/RecSys course/Demo 1/jabril-movie-ratings.csv", newline='') as csvfile:
    ratings_reader = csv.DictReader(csvfile)
    for row in ratings_reader:
        if (row['ratings'] != "") and (float(row['ratings']) > 0) and (float(row['ratings']) < 6):
            jabril_rating_dict.update({int(row['item']): float(row['ratings'])})

print("Rating dictionaries assembled!")
print("Sanity check:")
print("\tJabril's rating for 1197 (The Princess Bride) is " + str(jabril_rating_dict[1197]))

num_recs = 10  # Number of recommendations to generate
user_user = UserUser(15, min_nbrs=3)  # Minimum (3) and maximum (15) number of neighbors to consider
algo = Recommender.adapt(user_user)
algo.fit(data.ratings)

print("Set up a User-User algorithm!")

# Here, -1 tells it that it's not an existing user in the set, that we're giving new ratings
jabril_recs = algo.recommend(-1, num_recs, ratings=pd.Series(jabril_rating_dict))
joined_data = jabril_recs.join(data.movies['genres'], on='item')
joined_data = joined_data.join(data.movies['title'], on='item')
print("\n\nRECOMMENDED FOR JABRIL:")
print(joined_data)