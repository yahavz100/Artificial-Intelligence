# Yahav Zarfati 313163255

import sys
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances

USERID = 'userId'
MOVIEID = 'movieId'
TITLE = 'title'
RATING = 'rating'


class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []
        self.movieIdToMovieTitle: dict = dict()
        self.topRatings: list = list()

    def create_fake_user(self, rating):
        "*** YOUR CODE HERE ***"

        # Create a new dictionary representing fake user with movies he rated
        fakeUser: dict = {USERID: [283238, 283238, 283238, 283238, 283238], 'movieId': [33794, 153, 592, 595, 596],
            'rating': [5.0, 4.5, 5.0, 1.5, 0.5]}
        fakeUserDataFrame = pd.DataFrame(fakeUser)

        # Add user and his selections to ratings
        temp = pd.concat([rating, fakeUserDataFrame], ignore_index=True)
        rating = temp
        return rating

    def create_user_based_matrix(self, data):
        ratings = data[0]

        # for adding fake user
        ratings = self.create_fake_user(ratings)

        "*** YOUR CODE HERE ***"
        self.movieIdToMovieTitle = data[1]

        # Merge csv files, arrange table as userId's and movie titles by ratings
        ratings_pd = data[1].merge(ratings, on=MOVIEID)
        ratings_pd = ratings_pd.pivot_table(index=USERID, columns=TITLE, values=RATING)
        ratings = ratings_pd.to_numpy()

        # Calculate each row mean
        meanUserRating = ratings_pd.mean(axis=1).to_numpy().reshape(-1, 1)

        # Calculate the difference matrix by subtracting from each cell its row mean
        zeroMeanRatings = (ratings - meanUserRating)
        zeroMeanRatings[np.isnan(zeroMeanRatings)] = 0

        # Calculate the user similarity using cosine metric
        userSimilarity = 1 - pairwise_distances(zeroMeanRatings, metric='cosine')

        # Calculate the prediction matrix based on users similarity
        pred = meanUserRating + userSimilarity.dot(zeroMeanRatings) / np.array([np.abs(userSimilarity).sum(axis=1)]).T

        # Remove already rated movies, and create dataframe with userId's and movie titles
        removeSeenMovies(pred, ratings_pd)
        pred = pd.DataFrame(data=pred, index=[userId for userId in ratings_pd.index],
                            columns=[col for col in ratings_pd.columns])

        self.user_based_matrix = pred

    def create_item_based_matrix(self, data):
        "*** YOUR CODE HERE ***"
        self.movieIdToMovieTitle = data[1]

        # Merge csv files, arrange table as userId's and movie titles by ratings
        ratings_pd = data[1].merge(data[0], on=MOVIEID)
        ratings_pd = ratings_pd.pivot_table(index=USERID, columns=TITLE, values=RATING)
        ratings = ratings_pd.to_numpy()

        # Calculate each row mean
        meanUserRating = ratings_pd.mean(axis=1).to_numpy().reshape(-1, 1)

        # Calculate the difference matrix by subtracting from each cell its row mean
        ratingsDifference = (ratings - meanUserRating)
        ratingsDifference[np.isnan(ratingsDifference)] = 0

        ratingItem = ratingsDifference
        ratingItem[np.isnan(ratingItem)] = 0

        # Calculate the user similarity using cosine metric
        itemSimilarity = 1 - pairwise_distances(ratingItem.T, metric='cosine')

        # Calculate the prediction matrix based on users similarity
        pred = meanUserRating + ratingItem.dot(itemSimilarity) / np.array([np.abs(itemSimilarity).sum(axis=1)])

        # Remove already rated movies, and create dataframe with userId's and movie titles
        removeSeenMovies(pred, ratings_pd)
        pred = pd.DataFrame(data=pred, index=[userId for userId in ratings_pd.index],
                            columns=[col for col in ratings_pd.columns])

        self.item_based_metrix = pred

    def predict_movies(self, user_id, k, is_user_based=True):
        "*** YOUR CODE HERE ***"

        # Check if requested prediction based on items or users
        if is_user_based == True:
            predictionMatrix = self.user_based_matrix
        else:
            predictionMatrix = self.item_based_metrix

        # Find user row and return a list of top K movies
        userRow = predictionMatrix.loc[int(user_id)]
        topMovies = userRow.nlargest(k).index.values

        # Cache list of the ratings respectively
        self.topRatings = userRow.nlargest(k).values

        return list(topMovies)

    def getUserBasedMatrix(self):
        return self.user_based_matrix

    def getItemBasedMatrix(self):
        return self.item_based_metrix


def removeSeenMovies(prediction, source):
    """
    param: prediction, prediction matrix
    param: source, original matrix
    return: Does not return a value, but CHANGES prediction matrix cells
    """
    # Iterate over all values in source matrix, for each cell[i,j] which value in source matrix !=0,
    # set 0 in prediction cell[i,j]
    source = source.fillna(0).to_numpy()
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            value = source[i, j]
            # Check if value is not 0 (user rated it before)
            if value != 0:
                prediction[i, j] = 0

