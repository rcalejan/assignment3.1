'''
This holds the file where we're going to dump the dictionary 
holding all the delicious inverted index information from a directory
'''

import json
import pickle
import os
import datetime
import math

###################################################################
#                          Constants                              #
###################################################################
current_file = 1
NUMBER_OF_FILES = 55393
PAGE_RANK_FACTOR = 1000

###################################################################
#                          Functions                              #
###################################################################

def dumpToPickle(index: dict[str, list], number: int) -> None:
    """Dump dictionary to pickle file."""
    with open(f'dumps/dump{number}.pickle', 'wb') as dump_file:
        pickle.dump(index, dump_file, pickle.HIGHEST_PROTOCOL)


def postingToList(posting: object) -> list:
    """Return posting as list for serialization"""
    return [posting.docID, posting.freq, posting.posList, posting.title, posting.bold, posting.header]


def merge() -> None:
    """Merges all dumped pickle files into 5 masterIndexes."""

    '''
    RANGES
    1: a-e
    2: f-j
    3: k-o
    4: p-s
    5: t-z + other
    '''

    for master_index in range(5):
        print(f"{datetime.datetime.now()}: Master Index: {master_index}")
        new_partial_index = {}
        for index, file in enumerate(os.listdir('./dumps')):
            print(f'\t{datetime.datetime.now()}: Dump: {index}')
            file = os.path.join('./dumps', file)
            dump = open(file, 'rb')
            partial_index = pickle.load(dump)
            match master_index:
                case 0:
                    for k, v in partial_index.items():
                        if k[0] >= 'a' and k[0] <= 'e':
                            if k in new_partial_index:
                                new_partial_index[k].extend(v)
                            else:
                                new_partial_index[k] = v
                case 1:
                    for k, v in partial_index.items():
                        if k[0] >= 'f' and k[0] <= 'j':
                            if k in new_partial_index:
                                new_partial_index[k].extend(v)
                            else:
                                new_partial_index[k] = v
                case 2:
                    for k, v in partial_index.items():
                        if k[0] >= 'k' and k[0] <= 'o':
                            if k in new_partial_index:
                                new_partial_index[k].extend(v)
                            else:
                                new_partial_index[k] = v
                case 3:
                    for k, v in partial_index.items():
                        if k[0] >= 'p' and k[0] <= 's':
                            if k in new_partial_index:
                                new_partial_index[k].extend(v)
                            else:
                                new_partial_index[k] = v
                case 4:
                    for k, v in partial_index.items():
                        if k[0] > 's':
                            if k in new_partial_index:
                                new_partial_index[k].extend(v)
                            else:
                                new_partial_index[k] = v

        with open(f'./masterIndex/index{master_index}.json', 'w') as json_file:
            print(f"{datetime.datetime.now()} Started Dump")
            json.dump(new_partial_index, json_file, default=postingToList)
            print(f"{datetime.datetime.now()} Ended Dump")


def createFinalIndex() -> None:
    """Merge all master indexes into final Indexes while creating location indexes for each final index."""
    token_idf_index = {}
    # Load PageRank dictinoary
    print("Loading PageRanks Index...")
    with open(f'./pageRanks/pageRankIndex.json', 'r') as json_file:
        page_rank_index = json.load(json_file)
    print("\tFinished")
    for i in range(5):
        print(f"Working on index: {i}")
        with open(f'./masterIndex/index{i}.json', 'r') as json_file:
            partial_index = json.load(json_file)
        print("\tLoaded partial index from masterIndex")
        print(f"Updating token_idf_index...")
        for token, postings_list in partial_index.items():
            token_idf_index[token] = math.log(NUMBER_OF_FILES/len(postings_list), 10)
        print("\tCreating final_index and index_index...")
        with open(f'finalIndex/index{i}.json', 'w') as final_index_file:
            index_dict = {}
            for token, postings_list in partial_index.items():
                postings_list = sorted(postings_list, key=lambda posting: posting[1] + PAGE_RANK_FACTOR * page_rank_index[str(posting[0])])
                start_postion = final_index_file.tell()
                json.dump(postings_list[:100], final_index_file)
                first_hundred_end_position = final_index_file.tell()
                json.dump(postings_list[100:], final_index_file)
                real_end_position = final_index_file.tell()
                index_dict[token] = (start_postion, first_hundred_end_position, real_end_position)
        print("\tFinished")
        with open(f'./indexIndex/index{i}.pickle', 'wb') as index_file:
            pickle.dump(index_dict, index_file, pickle.HIGHEST_PROTOCOL)
        print("\tSaved index_index")
    with open(f'./tokenIdfIndex/tokenIdf.pickle', 'wb') as index_file:
        pickle.dump(token_idf_index, index_file, pickle.HIGHEST_PROTOCOL)
    print("\tComplete")
