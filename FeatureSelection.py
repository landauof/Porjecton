# imports
import pandas
from collections import Counter


# vars
input_path = 'MFDCA-DATA/FraudedFeatureOutputs/output'
feature_selection_output_path =\
    'MFDCA-DATA/FraudedFeatureSelectedOutputs/output'

# Magic


def select_features(features_number, method=1, user_number=0):
    global number_of_features, number_of_user
    number_of_user = user_number
    number_of_features = features_number
    methods = {1: most_common_ngrams}
    return methods[method]()


def load_data_of_user():
    """
    Load data
    :param:
    :return: user data
    """
    df = pandas.read_csv(input_path+str(number_of_user)+'.csv')
    return df


def most_common_ngrams():
    data_frame = load_data_of_user()

    col_num = data_frame.shape[1]-1

    array = data_frame.values
    X = array[:, 0:col_num]    # number of occurrences of each feature
    Y = array[:, col_num]      # Class
    # feature selection

    score = [0] * col_num
    for i in X:
        score = score + i

    feature_dictionary = Counter(dict(zip(data_frame.columns, score)))    # map in order to easily find top features
    top_features = feature_dictionary.most_common(number_of_features)
    top_features = dict(top_features)
    top_features = top_features.keys()

    for col in data_frame.columns:
        if col not in top_features:
            data_frame.pop(col)

    data_frame['Class'] = Y

    data_frame.to_csv(feature_selection_output_path + str(number_of_user)+'.csv')
    print("selection")


# for ind in range(40):
#    select_features(220, 1, ind)
#    print("done user {}".format(ind))
