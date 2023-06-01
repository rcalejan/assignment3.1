"""Files contains function definitions for merging indexes together and creating indexes."""

###################################################################
#                           Imports                               #
###################################################################
import datetime
import json
import pickle
import os
import math

###################################################################
#                          Constants                              #
###################################################################

NUMBER_OF_FILES = 55393
PAGE_RANK_FACTOR = 1000

###################################################################
#                          Functions                              #
###################################################################

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
    # Iterate through 5 master indexes
    for master_index in range(5):
        print(f"{datetime.datetime.now()}: Master Index: {master_index+1}")
        # Initialize empty index
        new_partial_index = {}

        # Iterate through each dump
        for index, file in enumerate(os.listdir('./dumps')):
            print(f'\t{datetime.datetime.now()}: Dump: {index+1}')
            
            # Load Dump
            file = os.path.join('./dumps', file)
            dump = open(file, 'rb')
            partial_index = pickle.load(dump)

            # Only add tokens in dumps that correspond to correct master index
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

        # Save combined index to file
        with open(f'./masterIndex/index{master_index}.json', 'w') as json_file:
            print(f"{datetime.datetime.now()} Started Dump")
            json.dump(new_partial_index, json_file, default=postingToList)
            print(f"{datetime.datetime.now()} Ended Dump")


def createFinalIndex() -> None:
    """Merge all master indexes into final Indexes while creating location indexes for each final index."""
    # Intialize token idf index
    token_idf_index = {}

    # Load PageRank dictinoary
    print("Loading PageRanks Index...")
    with open(f'./pageRanks/pageRankIndex.json', 'r') as json_file:
        page_rank_index = json.load(json_file)
    print("\tFinished")

    # Iterate through the 5 master indexes
    for i in range(5):
        # Load Master Index corresponding to final index
        print(f"Working on index: {i+1}")
        with open(f'./masterIndex/index{i}.json', 'r') as json_file:
            partial_index = json.load(json_file)
        print("\tLoaded partial index from masterIndex")

        # Update idf token index with each tokens idf
        print(f"\tUpdating token_idf_index...")
        for token, postings_list in partial_index.items():
            token_idf_index[token] = math.log(NUMBER_OF_FILES/len(postings_list), 10)
        print("\tCreating final_index and index_index...")

        # Create Final Index
        with open(f'finalIndex/index{i}.json', 'w') as final_index_file:
            # Initialize empty Index
            index_dict = {}

            # Iterate through postings in master index
            for token, postings_list in partial_index.items():
                # Sort Postings by score that takes into account frequency of token and page rank of document
                postings_list = sorted(postings_list, key=lambda posting: posting[1] + PAGE_RANK_FACTOR * page_rank_index[str(posting[0])], reverse=True)

                # Keep track of start position in file
                start_postion = final_index_file.tell()
                # Dump first hudnred postings
                json.dump(postings_list[:100], final_index_file)
                # Keep track of end of first hundred documents of term
                first_hundred_end_position = final_index_file.tell()
                # Dump rest of postings
                json.dump(postings_list[100:], final_index_file)
                # Keep track of end position of all postings
                real_end_position = final_index_file.tell()
                # Save position information to location index
                index_dict[token] = (start_postion, first_hundred_end_position, real_end_position)
        print("\tFinished")

        # Save Location Index to file
        with open(f'./indexIndex/index{i}.pickle', 'wb') as index_file:
            pickle.dump(index_dict, index_file, pickle.HIGHEST_PROTOCOL)
        print("\tSaved index_index")
    # Save Token IDF Index to file
    with open(f'./tokenIdfIndex/tokenIdf.pickle', 'wb') as index_file:
        pickle.dump(token_idf_index, index_file, pickle.HIGHEST_PROTOCOL)
    print("\tComplete")

