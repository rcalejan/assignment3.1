'''
This holds the file where we're going to dump the dictionary 
holding all the delicious inverted index information from a directory
'''

###################################################################
#                           Imports                               #
###################################################################
import json
import pickle
import os
import datetime
import math


###################################################################
#                          Functions                              #
###################################################################

def dumpToPickle(index: dict[str, list], number: int) -> None:
    """Dump dictionary to pickle file."""
    with open(f'dumps/dump{number}.pickle', 'wb') as dump_file:
        pickle.dump(index, dump_file, pickle.HIGHEST_PROTOCOL)





def savePageRanks(index: dict[str, list]) -> None:
    """Dump hash dictionary to json file."""
    with open(f'./pageRanks/pageRankIndex.json', 'w') as json_file:
        print(f"{datetime.datetime.now()} Started Saving Page Ranks")
        json.dump(index, json_file)
        print(f"{datetime.datetime.now()} Finished Saving Page Ranks")


def saveHashes(index: dict[str, list]) -> None:
    """Dump hash dictionary to json file."""
    with open(f'./docHashes/hashIndex.json', 'w') as json_file:
        print(f"{datetime.datetime.now()} Started Saving Hash Index")
        json.dump(index, json_file)
        print(f"{datetime.datetime.now()} Finished Saving Hash Index")


def load_index(index_num: int) -> dict[str, int]:
    """Loads index of character start and end locations for each token in file."""
    with open(f'./indexIndex/index{index_num}.pickle', 'rb') as pickle_data:
        location_index = pickle.load(pickle_data)
    return location_index


def load_page_rank_index() -> dict[int, int]:
    """Loads index of Page Ranks"""
    print("Loading PageRanks Index...")
    with open(f'./pageRanks/pageRankIndex.json', 'r') as json_file:
        page_rank_index = json.load(json_file)
    print("\tFinished")
    return page_rank_index


def load_token_idf_index() -> dict[int, int]:
    """Loads index of token idf index."""
    print("Loading tokenIdfIndex Index...")
    with open(f'./tokenIdfIndex/tokenIdf.pickle', 'rb') as pickle_file:
        token_idf_index = pickle.load(pickle_file)
    print("\tFinished")
    return token_idf_index


