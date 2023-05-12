'''
This holds the file where we're going to dump the dictionary 
holding all the delicious inverted index information from a directory
'''

import json
import pickle
from json import JSONEncoder
from index import Posting
import os
current_file = 1

def dumpToJson(index, number):
    with open(f'dumps/dump{number}.pickle', 'wb') as dump_file:
        pickle.dump(index, dump_file, pickle.HIGHEST_PROTOCOL)


def merge():
    '''
    RANGES
    1: a-e
    2: f-j
    3: k-o
    4: p-s
    5: t-z + other
    '''
    result = list()
    file1 = open('masterIndex/index1.pickle', 'wb')
    file1.close()
    file2 = open('masterIndex/index2.pickle', 'wb')
    file2.close()
    file3 = open('masterIndex/index3.pickle', 'wb')
    file3.close()
    file4 = open('masterIndex/index4.pickle', 'wb')
    file4.close()
    file5 = open('masterIndex/index5.pickle', 'wb')
    file5.close()
    for file in os.listdir('./dumps'):
        file = os.path.join('./dumps', file)
        dump = open(file, 'rb')
        partial_index = pickle.load(dump)
        partial_1 = dict()
        partial_2 = dict()
        partial_3 = dict()
        partial_4 = dict()
        partial_5 = dict()
        for k, v in partial_index.items():
            if k[0] >= 'a' and k[0] <= 'e':
                partial_1[k] = v
            elif k[0] >= 'f' and k[0] <= 'j':
                partial_2[k] = v
            elif k[0] >= 'k' and k[0] <= 'o':
                partial_3[k] = v
            elif k[0] >= 'p' and k[0] <= 's':
                partial_4[k] = v
            else:
                partial_5[k] = v
        del partial_index

        pickle_1_r = open('./masterIndex/index1.pickle', 'rb')
        try:
            index1 = pickle.load(pickle_1_r)
            index1.update(partial_1)
        except EOFError:
            index1 = dict()
        pickle_1_r.close()

        pickle_1_w = open('./masterIndex/index1.pickle', 'wb')
        pickle.dump(index1, pickle_1_w, pickle.HIGHEST_PROTOCOL)
        del index1

        pickle_2_r = open('./masterIndex/index2.pickle', 'rb')
        try:
            index2 = pickle.load(pickle_2_r)
            index2.update(partial_2)
        except EOFError:
            index2 = dict()
        pickle_2_r.close()

        pickle_2_w = open('./masterIndex/index2.pickle', 'wb')
        pickle.dump(index2, pickle_2_w, pickle.HIGHEST_PROTOCOL)
        del index2

        pickle_3_r = open('./masterIndex/index3.pickle', 'rb')
        try:
            index3 = pickle.load(pickle_3_r)
            index3.update(partial_3)
        except EOFError:
            index3 = dict()
        pickle_3_r.close()

        pickle_3_w = open('./masterIndex/index3.pickle', 'wb')
        pickle.dump(index3, pickle_3_w, pickle.HIGHEST_PROTOCOL)
        del index3

        pickle_4_r = open('./masterIndex/index4.pickle', 'rb')
        try:
            index4 = pickle.load(pickle_4_r)
            index4.update(partial_4)
        except EOFError:
            index4 = dict()
        pickle_4_r.close()

        pickle_4_w = open('./masterIndex/index4.pickle', 'wb')
        pickle.dump(index4, pickle_4_w, pickle.HIGHEST_PROTOCOL)
        del index4

        pickle_5_r = open('./masterIndex/index5.pickle', 'rb')
        try:
            index5 = pickle.load(pickle_5_r)
            index5.update(partial_5)
        except EOFError:
            index5 = dict()
        pickle_5_r.close()

        pickle_5_w = open('./masterIndex/index5.pickle', 'wb')
        pickle.dump(index5, pickle_5_w, pickle.HIGHEST_PROTOCOL)
        del index5