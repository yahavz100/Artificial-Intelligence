# Yahav Zarfati 313163255ב

from sklearn.metrics import mean_squared_error
from math import sqrt
# Import Pandas
import pandas as pd
import math

USERID = 'userId'
MOVIEID = 'movieId'
TITLE = 'title'
RATING = 'rating'
K = 10
HIT = 4
PRECISION10 = 0
ARHR = 1
RMSE = 2


def precision_10(test_set, cf, is_user_based=True):
    "*** YOUR CODE HERE ***"

    val = calculatePrecision(test_set, cf, is_user_based, PRECISION10)
    print("Precision_k: " + str(val))


def ARHA(test_set, cf, is_user_based=True):
    "*** YOUR CODE HERE ***"

    val = calculatePrecision(test_set, cf, is_user_based, ARHR)
    print("ARHR: " + str(val))


def RSME(test_set, cf, is_user_based=True):
    "*** YOUR CODE HERE ***"

    val = calculatePrecision(test_set, cf, is_user_based, RMSE)
    print("RMSE: " + str(val))


def calculatePrecision(test_set, cf, is_user_based, method=ARHR):
    """
    param: test_set, test data
    param: cf, collaborative filtering object with user/item based matrix
    param: is_user_based, bool True - user based, False - item based
    param: method, int 0 - Precision@10, 1 - ARHR, 2 - RMSE
    return int, precision value
    Function calculates the precision of cf using different methods
    """
    data = cf.movieIdToMovieTitle
    precision = 0

    # Merge csv files, arrange table as userId's and movie titles by ratings
    test_set_dataframe = pd.DataFrame(test_set).merge(data, on=MOVIEID)
    test_set_dataframe = test_set_dataframe.pivot_table(index=USERID, columns=TITLE, values=RATING)

    if method == PRECISION10 or method == ARHR:

        # List all userId's in test set
        usersIds = [x for x in test_set_dataframe.index]
        userToTopMovies: dict = dict()

        # Map each user id to its top K movies predicted using dictionary
        for user in usersIds:
            userToTopMovies[user] = cf.predict_movies(user, K, is_user_based)

        # For each user, check each movie in user's top K list, also, if it's a hit
        for item in userToTopMovies.items():
            userTopMovies = item[1]
            userId = item[0]

            # Check each user's top movie
            for movie in userTopMovies:

                test_set_rating = test_set_dataframe[movie][userId]

                # Check if it's rated as a HIT, then sum all hits
                if test_set_rating >= HIT:

                    # Check which precision method was chosen
                    # If its PRECISION@10 sum 1/K
                    if method == PRECISION10:
                        precision += (1 / K)

                    # Else if its ARHR sum 1/pos
                    elif method == ARHR:
                        pos = userTopMovies.index(movie)
                        precision += (1 / (pos + 1))
                # Check if it's the default precision - RMSE
                if method == RMSE:
                    pos = userTopMovies.index(movie)
                    predicted_rating = cf.topRatings[pos]
                    precision += (pow((test_set_rating - predicted_rating), 2))

        # Divide number of hits with number of users
        precision = precision / len(usersIds)

    # Else, it's precision - RMSE
    else:
        test_set_numpy = test_set_dataframe.to_numpy()

        # Check if its user/item based and user the correct matrix respectively
        if is_user_based == True:
            ratings = cf.getUserBasedMatrix().to_numpy()
        else:
            ratings = cf.getItemBasedMatrix().to_numpy()

        # Iterate over test matrix
        for i in range(test_set_numpy.shape[0]):
            for j in range(test_set_numpy.shape[1]):
                test_value = test_set_numpy[i, j]

                # Check if value in cell [i,j] is nan, if it is continue
                isNan = math.isnan(test_value)
                if isNan == True:
                    continue
                # Else, its not nan, add it to precision
                predicted_rating = ratings[i][j]
                precision += (pow((test_value - predicted_rating), 2))

        # Divide number of hits with test size and squared the result
        precision = precision / len(test_set)
        precision = sqrt(precision)
    return precision
