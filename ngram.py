"""File holds functions for NGram operations."""
###################################################################
#                           Imports                               #
###################################################################

from collections import defaultdict
import re
from porter2stemmer import Porter2Stemmer

###################################################################
#                Global Variabls and Constants                    #
###################################################################

stemmer = Porter2Stemmer()

###################################################################
#                          Functions                              #
###################################################################

def getPostingsForNGram(postings, distance, first_word, second_word):
    """Returns pairs of postings that contain both words and has the second_word in the second posting
        a specified 'distance' after the first_word in the first posting."""
    # Sort postings for each token so we can combine them faster
    first_word_postings = sorted(postings[first_word], key= lambda posting: posting[0])
    second_word_postings=  sorted(postings[second_word], key= lambda posting: posting[0])

    # Initialize postion of posting in postings
    first_pos = 0
    second_pos = 0

    # Initialize current docID of current postings
    first_id = first_word_postings[first_pos][0]
    second_id = second_word_postings[second_pos][0]

    # Find max_id to continue checking under
    max_id = min(first_word_postings[-1][0], second_word_postings[-1][0])

    # Initialize postings to return
    n_gram_postings = []

    # Iterate through postings in order of DocID
    # If either DocID is greater than the max doc ID of the other then stop loop
    while first_id < max_id and second_id < max_id:
        # Increment second docID if it's smaller than the current first DocID
        if second_id < first_id:
            second_pos += 1
            second_id = second_word_postings[second_pos][0]

        # Increment first docID if it's smaller than the current second DocID
        elif first_id < second_id:
            first_pos += 1
            first_id = first_word_postings[first_pos][0]
        
        # If DocID's are equal (in both postings list)
        else:
            # Check if second document contains the second word 'distance' amount 
            # after the location of the first word in the first document

            # Iterate thorugh each position in posting for first_word
            for first_word_position in first_word_postings[first_pos][2]:
                # Find needed position of second_word by adding distance to first_word
                second_word_position = first_word_position + distance
                # See if needed position is in posting for second_word
                if second_word_position in second_word_postings[second_pos][2]:
                    # If so, append pair to postings
                    n_gram_postings.append((first_word_postings[first_pos], second_word_postings[second_pos]))
                    break
            # Increment both postings list current docID's
            first_pos += 1
            first_id = first_word_postings[first_pos][0]
            second_pos += 1
            second_id = second_word_postings[second_pos][0]
    return n_gram_postings


def getNGramPostings(nGrams, postings):
    """Get postings for each n gram"""
    nGramPostings = {}
    if nGrams:
        # Iterate through NGrams
        for nGram in nGrams:
            # Get first word, second word, and distance between in query
            distance, first_word, second_word = nGram
            # Check if both words are in postings
            if first_word in postings and second_word in postings:
                # Get and add postings to nGramPostings
                nGramPostings[nGram] = getPostingsForNGram(postings, distance, first_word, second_word)
    return nGramPostings


def NGramTokenizer(query: str) -> list[tuple[int, str, str]]:
    """Gets all N Grams in query.  NGrams are returned in a list
        where each item in the list is a tuple where the first item
        is the distance between the first and second words, the second
        item is the first word, and the second item is the second word.

        For Example: in the query "Computer Science At UCI", the function
        would return [(3, 'computer', 'uci')]"""
    query = query.lower()
    pattern = re.compile(r'\w+')
    tokens = pattern.findall(query)
    porterized = map(stemmer.stem, tokens)
    tokens = list(porterized)
    num_tokens = len(tokens)
    nGrams = []
    for i in range(3, num_tokens+1):
        for index in range(num_tokens-i):
            nGrams.append((i, tokens[index], tokens[index+i]))
    return nGrams