import data as da
import collaborative_filtering as cf
import evaluation as ev
import pandas as pd
from datetime import datetime
import time


# Load data
movies = pd.read_csv('data/movies_subset.csv', low_memory=False)
rating = pd.read_csv('data/ratings.csv', low_memory=False)
test_set = pd.read_csv('data/test.csv', low_memory=False)

cf1 = cf.collaborative_filtering()


# PART 1 - DATA
def analysis(data):
    da.watch_data_info(data)
    da.print_data(data)
    da.plot_data(data)


# PART 2 - COLLABORATING FILTERING RECOMMENDATION SYSTEM
def collaborative_filtering_rec(data, user_based=True):
    global cf1
    if user_based:
        cf1.create_user_based_matrix(data)
    else:
        cf1.create_item_based_matrix(data)
    result = cf1.predict_movies('283225', 5, user_based)
    print(result)
    fake_user_test = cf1.predict_movies(user_id='283238', k=5)
    print(fake_user_test)


# PART 3 - EVALUATION
def evaluate_rec(user_based=True):
    ev.precision_10(test_set, cf1, user_based)
    ev.ARHA(test_set, cf1, user_based)
    ev.RSME(test_set, cf1, user_based)


def main():
    print('main start at', datetime.now().strftime('%H:%M:%S'))
    start_time = time.time()
    analysis((rating, movies))
    print(f'Checkpoint 1 runtime: {round((time.time() - start_time), 4)} seconds')
    collaborative_filtering_rec((rating, movies), user_based=True)
    print(f'Checkpoint 2 runtime: {round((time.time() - start_time), 4)} seconds')
    evaluate_rec(user_based=True)
    print(f'Checkpoint 3 runtime: {round((time.time() - start_time), 4)} seconds')
    collaborative_filtering_rec((rating, movies), user_based=False)
    print(f'Checkpoint 4 runtime: {round((time.time() - start_time), 4)} seconds')
    evaluate_rec(user_based=False)
    print(f'Total runtime: {round((time.time() - start_time), 4)} seconds')
    print('main end at', datetime.now().strftime('%H:%M:%S'))


if __name__ == '__main__':
    main()
