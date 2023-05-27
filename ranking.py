import index as indexer
from collections import defaultdict
import pickle
import os
from glob import glob
import index
import json
import orjson
from porter2stemmer import Porter2Stemmer
from index import Posting
import math
import time
'''
This is where the ranking algorithm goes.
Look into tf-idf ranking
'''

###################################################################
#                          Constants                              #
###################################################################

NUMBER_OF_FILES = 55393
current_id = 0
file_finder = dict()
stemmer = Porter2Stemmer()
stop_words = [
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'any', 'are', "aren't", 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could',
    "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each',
    'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd",
    "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i',
    "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me',
    'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's",
    'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this',
    'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're",
    "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
    "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're",
    "you've", 'your', 'yours', 'yourself', 'yourselves'
]

DOCUMENT_PATHS = [file for dir in os.walk('developer/DEV') for file in glob(os.path.join(dir[0], '*.json'))]

BOLD_VALUE = 5
TITLE_VALUE = 20
HEADER_VALUE = 10

###################################################################
#                          Functions                              #
###################################################################

def load_index(index_num: int) -> dict[str, int]:
    """Loads index of character start and end locations for each token in file."""
    with open(f'./indexIndex/index{index_num}.pickle', 'rb') as pickle_data:
        location_index = pickle.load(pickle_data)
    return location_index


def start_up() -> list[dict[str, int]]:
    """Loads each location index and returns them."""
    print("Loading Location Indexes from indexIndex...")
    location_index_0 = load_index(0)
    print("\tLoaded location index 1 of 5")
    location_index_1 = load_index(1)
    print("\tLoaded location index 2 of 5")
    location_index_2 = load_index(2)
    print("\tLoaded location index 3 of 5")
    location_index_3 = load_index(3)
    print("\tLoaded location index 4 of 5")
    location_index_4 = load_index(4)
    print("\tLoaded location index 5 of 5")
    return location_index_0, location_index_1, location_index_2, location_index_3, location_index_4


def askUser()->str:
    """Prompts User for query and returns input string"""
    print("==========  Welcome to GHoogle!  ==========\n")
    query = input("What do you want to know? ('quit' to exit): ")
    return query


def getPostings(tokens: list) -> dict[list[list]]:
    """Gets all postings for every token in tokens."""
    all_postings = defaultdict(list)  # Initialize Dictionary
    for token in set(tokens):
        if token not in stop_words:  # If token is a stop word then don't look for it's posting
            for new_posting in findPostings(token):
                all_postings[token].append(new_posting)
    return all_postings


def findPostings(token: str) -> list:
    """Finds postings in file and returns it as a list."""

    # Find which final index and location index to use
    if token[0] >= 'a' and token[0] <= 'e':
        index_to_use = location_index_0
        index_num = 0
    elif token[0] >= 'f' and token[0] <= 'j':
        index_to_use = location_index_1
        index_num = 1
    elif token[0] >= 'k' and token[0] <= 'o':
        index_to_use = location_index_2
        index_num = 2
    elif token[0] >= 'p' and token[0] <= 's':
        index_to_use = location_index_3
        index_num = 3
    else:
        index_to_use = location_index_4
        index_num = 4

    # Find start and end positions of token in file
    try:
        start_position, end_position = index_to_use[token]  # Get start and end position of json list in file
    except KeyError:
        return []    # if token wasn't found return empty list
    with open(f'./finalIndex/index{index_num}.json', 'r') as final_index:
        final_index.seek(start_position)    # Jump to file start position
        posting_data = final_index.read(end_position-start_position)    # Read object string
    postings = orjson.loads(posting_data)    # Load json string into dictionary
    return postings[token]


def computeQueryFrequencies(tokens: list[str]) -> dict[str, int]:
    """Compute frequency of word in query."""
    frequencies = defaultdict(int)
    for token in tokens:
        frequencies[token] += 1
    return frequencies


def calculateQueryNormalizedTfIdf(query_tokens: list[str], postings: list[list], frequencies: dict[str, int]) -> dict[str, int]:
    """Calculates normalized tf-idf for query tokens."""

    # Calculate tf-wt
    query_tf_wt = {}
    tokens = set(query_token for query_token in query_tokens if query_token in postings)
    for token in tokens:
        try:
            query_tf_wt[token] = 1 + math.log(frequencies[token], 10)
        except:
            print("error", frequencies[token])

    # Calcualte idf
    query_idf = {}
    for token in tokens:
        query_idf[token] = math.log(NUMBER_OF_FILES/sum([posting[1] for posting in postings[token]]))

    # Claculate tf-idf
    query_weights = {}
    for token in tokens:
        query_weights[token] = query_tf_wt[token] * query_idf[token]

    # Normalize tf-idf
    query_normalized_scores = {}
    query_normalization_length = math.sqrt(sum([i**2 for i in query_weights.values()]))
    for token, weight in query_weights.items():
        query_normalized_scores[token] = weight/query_normalization_length
    
    return query_normalized_scores


def calculateDocumentNormalizedTfWt(postings: list[list]) -> dict[str, int]:
    """Calcualte normalized tf scores for each document."""
    document_term_scores = defaultdict(dict)
    for token, posting in postings.items():
        for docID, freq, posList, title, bold, header in posting:
            # Calcualte tf-wt and add scores for if token was a title, bold, or header
            document_term_scores[docID][token] = 1 + math.log(freq) + title * TITLE_VALUE + bold * BOLD_VALUE + header * HEADER_VALUE
        # Normalize Values
        document_term_scores[docID][token] = document_term_scores[docID][token]/sum(document_term_scores[docID].values())

    return document_term_scores
    

def search(query: str) -> list[tuple[int, int]]:
    """Searches for documents that match query."""
    # Tokenize and stem query
    tokenized_query = indexer.tokenize(query)
    # Remove stop words
    query_tokens = [token for token in tokenized_query if token not in stop_words]
    print(f"Stemmed to: {query_tokens}")
    # Compute word frequencies of query
    frequencies = computeQueryFrequencies(query_tokens)
    # Get postings of all tokens
    postings = getPostings(query_tokens)
    # Calculate tf-idf of query
    query_normalized_weights = calculateQueryNormalizedTfIdf(query_tokens, postings, frequencies)
    # Calculate tf-wt of documents
    document_normalized_weights = calculateDocumentNormalizedTfWt(postings)
    # Calcualte cosine similarity values of each document
    document_scores = defaultdict(int)
    for document, token_normalized_scores in document_normalized_weights.items():
        for token, score in token_normalized_scores.items():
            document_scores[document] += score * query_normalized_weights[token]
    # Sort documents by score
    sorted_list_of_documents = sorted(document_scores.items(), key = lambda document: document[1], reverse=True)
    return sorted_list_of_documents


def getTopKUrls(documents: list[tuple[int, int]], k: int) -> list[tuple[int, str]]:
    """Return urls of the top k documents sorted by score."""
    urls = []
    for index, (documentid, score) in enumerate(documents):
        if index == k:
            break
        doc_path = DOCUMENT_PATHS[documentid]
        with open(doc_path, 'r') as file:
            data = json.load(file)
            urls.append(data)
    return urls

###################################################################
#                             Main                                #
###################################################################

if __name__ == '__main__':
    # Load location indexes into memory
    print("Starting Up")
    location_index_0, location_index_1, location_index_2, location_index_3, location_index_4 = start_up()
    print("Done")

    # Start search engine loop in command line
    while True:
        query = askUser()  # Prompt user for query
        print()
        if query.lower() in ['quit', 'exit', 'stop']:
            break
        start_time = time.time()
        sorted_postings = search(query)  # Get ordered documents
        urls = getTopKUrls(sorted_postings, 10) # Return Top 10 documents
        end_time = time.time()
        print()

        # Print Results
        print("================  Results  ================\n")
        for document_info in urls:
            print(document_info)
        print(f"\nFound {len(sorted_postings)} results in {end_time - start_time} seconds\n")
