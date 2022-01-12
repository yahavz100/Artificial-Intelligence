# Yahav Zarfati 313163255

import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

USERID = 'userId'
MOVIEID = 'movieId'
RATING = 'rating'

def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes,
        # non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a
        # dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data: tuple):
    "*** YOUR CODE HERE ***"

    ratingsCSVFile = data[0]

    # Count the number of unique users
    uniqueUsersCounter = len(ratingsCSVFile[USERID].unique())

    # Count the number of unique movies
    uniqueMoviesCounter = len(ratingsCSVFile[MOVIEID].unique())

    # Count the number of ratings
    ratingCounter = len(ratingsCSVFile[RATING])

    # Count number of minimum and maximum ratings given to a movie
    maximumNumberOfMovieRatings = findMinimumMaximum(category=MOVIEID, maxFlag=True, data=ratingsCSVFile,
                                                     returnDictionary=False)
    minimumNumberOfMovieRatings = findMinimumMaximum(category=MOVIEID, maxFlag=False, data=ratingsCSVFile,
                                                     returnDictionary=False)

    # Count number of minimum and maximum times user rated a movie
    maximumNumberOfUserRatings = findMinimumMaximum(category=USERID, maxFlag=True, data=ratingsCSVFile,
                                                    returnDictionary=False)
    minimumNumberOfUserRatings = findMinimumMaximum(category=USERID, maxFlag=False, data=ratingsCSVFile,
                                                    returnDictionary=False)

    print("Number of unique users:", uniqueUsersCounter)
    print("Number of unique movies:", uniqueMoviesCounter)
    print("Number of ratings:", ratingCounter)
    print("Maximum number of times movie was rated:", maximumNumberOfMovieRatings)
    print("Minimum number of times movie was rated:", minimumNumberOfMovieRatings)
    print("Maximum number of times user rated a movie:", maximumNumberOfUserRatings)
    print("Minimum number of times user rated a movie:", minimumNumberOfUserRatings)


def plot_data(data, plot=True):
    "*** YOUR CODE HERE ***"

    # Initialize a dictionary based on rating, (key: rating, value: counter of key rating)
    ratingsCSVFile = data[0]
    ratingsDictionary = findMinimumMaximum(category=RATING, maxFlag=False, data=ratingsCSVFile, returnDictionary=True)

    # Create a plot, X axis - rating, Y axis - counter of each key
    xData = ratingsDictionary.keys()
    yData = ratingsDictionary.values()
    ratingsPD = pd.DataFrame(yData, index=xData)
    ax = ratingsPD.plot.bar(rot=0)
    ax.set_title("Ratings Distribution")
    ax.set_xlabel("Ratings")
    ax.set_ylabel("Number of ratings")
    ax.legend(["Value"])

    # Check if plot flag is on, then show plot
    if plot == True:
        plt.show()
        plt.savefig('./plot/ratings.png')


def findMinimumMaximum(category: str, maxFlag: bool, data, returnDictionary: bool = False):
    """
    category: str, chosen category to find the value
    maxFlag: bool, True - return max value, False - return min value
    data: data
    returnDictionary: bool, True - return dictionary, False - do not return dictionary
    Function find minimum/maximum number of times a value has been found, from given data of given category
    If returnDictionary flag is on returns a dictionary, otherwise return min/max value
    return: (int, minimum/maximum number) or (dict, sorted dictionary by keys)
    """
    # Initialize dictionary counter
    dictionaryCounter: dict = dict()
    column = data[category]

    # For each value in category column, count
    for value in column:
        # Check if value already in dictionary, if it is, add to value 1, otherwise, its not in dictionary, add new key
        # and counter value 1
        if value in dictionaryCounter.keys():
            dictionaryCounter[value] += 1
        else:
            dictionaryCounter[value] = 1

    # Check if returnDictionary flag is on, if its on return min/max, otherwise, return sorted dictionary
    if returnDictionary == False:
        # Check if user want max or min
        if maxFlag == True:
            return max(dictionaryCounter.values())
        else:
            return min(dictionaryCounter.values())

    else:
        # Initialize new dictionary - sorted copy of dictionary counter
        sortedDictionary = dict()
        for i in sorted(dictionaryCounter):
            sortedDictionary[i] = dictionaryCounter[i]
        return sortedDictionary
