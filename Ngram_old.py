# Imports
from nltk import ngrams
from itertools import chain
from collections import Counter

# Variables
path = 'MFDCA-DATA/FraudedRawData/User'
n = 3


# Magic


def ngramize(commands, num):
    """
    :param commands:
    :param num:
    :return: num-grams of commands
    """
    return ngrams(commands.split(), num)


def read_file(user_number):
    """
    :param user_number:
    :return: commands from the input file
    """
    file = open(path+user_number, 'r')
    text = file.read()
    file.close()
    return text


def get_all_ngrams():
    """
    Iterate over all command files
    :return: a long list of num-grams
    """
    grams = ()
    for i in range(0, 40):
        text_i = read_file(str(i))
        curr_grams = ngramize(text_i, n)
        grams = chain(grams, curr_grams)
    return grams


def count_occurrences(gen):
    return Counter(gen)


def debug_all_users_at_once():
    all_grams = get_all_ngrams()    # all_grams = n-gram generator
    gram_and_occurrences = count_occurrences(all_grams)    # holds all grams and number of occurrences
    print(gram_and_occurrences)


def pre_process_of_commands(commands_as_string):
    """
    Turns string to list of 100 commands per string
    :param commands_as_string:
    :return: list of strings (blocks)
    """
    ans = []
    text = commands_as_string.split()
    for i in range(int(len(text)/100)):
        new_block = ''
        for j in range(100):
            new_block = new_block + text[i*100 + j] + ' '
        ans.append(new_block)
    return ans


def features_per_block(user):
    """
    :param user: the user number
    :return: list of counters of feature and occurrences
    """
    list_of_features_per_block = []
    block_list = pre_process_of_commands(read_file(str(user)))
    for block in block_list:
        block_grams = ngramize(block, n)
        block_features = count_occurrences(block_grams)
        list_of_features_per_block.append(block_features)
    return list_of_features_per_block


def per_user_feature_extraction():
    """
    Iterates on Users
    :return: List (users) of lists (blocks) of counters (n-grams : occurrences)
    """
    ans = []    # List of counters (dictionaries)
    for i in range(0, 40):
        gram_and_occurrences = features_per_block(i)
        ans.append(gram_and_occurrences)
    return ans


