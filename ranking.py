import index as indexer
from collections import defaultdict
import pickle
import os
from glob import glob
import json
import orjson
import math
import time
from porter2stemmer import Porter2Stemmer
import re

'''
This is where the ranking algorithm goes.
Look into tf-idf ranking
'''

###################################################################
#                          Constants                              #
###################################################################

NUMBER_OF_FILES = 55393
BOLD_VALUE = 5
TITLE_VALUE = 20
HEADER_VALUE = 10
QUERY_IDF_THRESHOLD = -0.75
DOCUMENT_PATHS = [file for dir in os.walk('developer/DEV') for file in glob(os.path.join(dir[0], '*.json'))]
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


###################################################################
#                          Functions                              #
###################################################################

def load_index(index_num: int) -> dict[str, int]:
    """Loads index of character start and end locations for each token in file."""
    with open(f'./indexIndex/index{index_num}.pickle', 'rb') as pickle_data:
        location_index = pickle.load(pickle_data)
    return location_index


def start_up() -> list[dict[str, int]]:
    """Loads each location, pageRank, and simHash indexes and returns them."""
    # Load Location Indexes
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

    # Load PageRank dictinoary
    print("Loading PageRanks Index...")
    with open(f'./pageRanks/pageRankIndex.json', 'r') as json_file:
        page_rank_index = json.load(json_file)
    print("\tFinished")
    
    # Load simHash dictionary
    print("Loading simHash Index...")
    with open(f'./docHashes/hashIndex.json', 'r') as json_file:
        sim_hash_index = json.load(json_file)
    print("\tFinished")
    
    return location_index_0, location_index_1, location_index_2, location_index_3, location_index_4, page_rank_index, sim_hash_index


def askUser()->str:
    """Prompts User for query and returns input string"""
    print("==========  Welcome to GHoogle!  ==========\n")
    query = input("What do you want to know? ('quit' to exit): ")
    return query


def getPostings(tokens: list) -> dict[list[list]]:
    """Gets all postings for every token and n gram."""
    all_postings = defaultdict(list)  # Initialize Dictionary
    for token in set(tokens):
        if token not in stop_words:  # If token is a stop word then don't look for it's posting
            for new_posting in findPostings(token):
                all_postings[token].append(new_posting)
    return all_postings


def getCombinedPostings(first_word, second_word, postings):
    """Returns postings that contain both words."""
    first_word_postings = postings[first_word]
    second_word_postings=  postings[second_word]
    common_docIDs = set([posting[0] for posting in first_word_postings]).intersection(set([posting[0] for posting in second_word_postings]))
    first_word_common_postings = [posting for posting in first_word_postings if posting[0] in common_docIDs]
    seecond_word_common_postings = [posting for posting in second_word_postings if posting[0] in common_docIDs]
    combinedPostings = []
    for i in range(len(first_word_common_postings)):
        combinedPostings.append((first_word_common_postings[i], seecond_word_common_postings[i]))
    return combinedPostings
        

def filterNGramPostings(postings, distance, first_word, second_word):
    filtered_postings = []
    for first_posting, second_posting in postings:
        for first__word_position in first_posting[2]:
            second_word_position = first__word_position + distance
            if second_word_position in second_posting[2]:
                filtered_postings.append((first_posting, second_posting))
                break
    return filtered_postings


def getNGramPostings(nGrams, postings):
    """Get postings for each n gram"""
    nGramPostings = {}
    if nGrams:
        for nGram in nGrams:
            distance, first_word, second_word = nGram
            combined_postings = getCombinedPostings(first_word, second_word, postings)
            nGramPostings[nGram] = filterNGramPostings(combined_postings, distance, first_word, second_word)
    return nGramPostings
        

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


def computeQueryFrequencies(tokens: list[str], nGrams: list[tuple[int, str, str]]) -> tuple[dict[str, int], dict[tuple, int]]:
    """Compute frequency of word in query."""
    frequencies = defaultdict(int)
    for token in tokens:
        frequencies[token] += 1
    nGramFrequencies = defaultdict(int)
    if nGrams:
        for nGram in nGrams:
            nGramFrequencies[nGram] += 1
    return frequencies, nGramFrequencies


def calculateQueryNormalizedTfIdf(query_tokens: list[str], postings: list[list], nGramPostings: dict[tuple, list], frequencies: dict[str, int], nGrams: list[tuple[int, str, str]], nGramFrequencies: dict[tuple, int]) -> dict[str, int]:
    """Calculates normalized tf-idf for query tokens."""

    # Calculate tf-wt
    query_tf_wt = {}
    tokens = set(query_token for query_token in query_tokens if query_token in postings and postings[query_token])
    for token in tokens:
        query_tf_wt[token] = 1 + math.log(frequencies[token], 10)

    # Calcualte idf
    query_idf = {}
    for token in tokens:
        try:
            idf = math.log(NUMBER_OF_FILES/len(postings[token]), 10)
            print(f"Token: {token} idf: {idf}")
            if idf > QUERY_IDF_THRESHOLD:
                query_idf[token] = idf
            else:
                print(f"Excluding query term \"{token}\" becasue it's idf was too small.")
        except:
            print(f"TOKEN: {token}")
            print(f"POSITNGS: {postings}")

    # Claculate tf-idf
    query_weights = {}
    for token in query_idf.keys():
        query_weights[token] = query_tf_wt[token] * query_idf[token]

    # Normalize tf-idf
    query_normalized_scores = {}
    query_normalization_length = math.sqrt(sum([i**2 for i in query_weights.values()]))
    for token, weight in query_weights.items():
        query_normalized_scores[token] = weight/query_normalization_length

    # Calculate tf-wt for nGrams
    if nGrams:
        query_tf_wt = {}
        nGrams = [nGram for nGram, posting_list in nGramPostings.items() if posting_list]
        for nGram in nGrams:
            query_tf_wt[nGram] = 1 + math.log(nGramFrequencies[nGram], 10)

        # Calcualte idf for nGrams
        query_idf = {}
        for nGram in nGrams:
            idf = math.log(NUMBER_OF_FILES/len(nGramPostings[nGram]), 10)
            print(f"Token: {nGram} idf: {idf}")
            if idf > QUERY_IDF_THRESHOLD:
                query_idf[nGram] = idf
            else:
                print(f"Excluding query term \"{nGram}\" becasue it's idf was too small.")

        # Claculate tf-idf for nGrams
        query_weights = {}
        for token in query_idf.keys():
            query_weights[token] = query_tf_wt[token] * query_idf[token]

        # Normalize tf-idf for nGrams
        query_nGram_normalized_scores = {}
        query_normalization_length = math.sqrt(sum([i**2 for i in query_weights.values()]))
        for token, weight in query_weights.items():
            query_nGram_normalized_scores[token] = weight/query_normalization_length
    else:
        query_nGram_normalized_scores = {}

    return query_normalized_scores, query_nGram_normalized_scores


def calculateDocumentNormalizedTfWt(postings: dict[str, list], nGramPostings: dict[tuple, list]) -> dict:
    """Calcualte normalized tf scores for each document."""
    document_term_scores = defaultdict(dict)
    for token, posting in postings.items():
        for docID, freq, posList, title, bold, header in posting:
            # Calcualte tf-wt and add scores for if token was a title, bold, or header
            document_term_scores[docID][token] = 1 + math.log(freq, 10) + title * TITLE_VALUE + bold * BOLD_VALUE + header * HEADER_VALUE
        # Normalize Values
        document_term_scores[docID][token] = document_term_scores[docID][token]/sum(document_term_scores[docID].values())
    
    # Same but for N Grams
    for nGram, first_second_postings in nGramPostings.items():
        for posting in first_second_postings:
            for docID, freq, posList, title, bold, header in posting:
                # Calcualte tf-wt and add scores for if nGram was a title, bold, or header
                document_term_scores[docID][nGram] = 1 + math.log(freq, 10) + title * TITLE_VALUE + bold * BOLD_VALUE + header * HEADER_VALUE
            # Normalize Values
            document_term_scores[docID][nGram] = document_term_scores[docID][nGram]/sum(document_term_scores[docID].values())
    return document_term_scores


def NGramTokenizer(query: str) -> list[tuple[int, str, str]]:
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

def is_similar(tokens: dict[str:int], url: str) -> bool:
    """Checks if file tokens are similar to a previously crawled page using sim_hash."""

    # Load hashed_pages dictionary
    filepath = os.path.join(os.path.dirname(__file__), 'hashed_pages.pickle')
    hashed_pages = load_dict_from_pickle(filepath)

    # Calcualte sim_hash value of the tokens using custom sim_hash function above
    hashed_value = sim_hash(tokens)
    hashed_value_bits = [(hashed_value >> bit) & 1 for bit in range(32 - 1, -1, -1)] # Convert sim_hash integer to binary list
    
    # Iterate through previously crawled urls and their sim_hash values
    for hashed_page_url,  value in hashed_pages.items():
        hashed_page_value = [(value >> bit) & 1 for bit in range(32 - 1, -1, -1)]  # Convert integer sim_hash value of previously crawled url to binary list
        
        # Count how many bits the previously crawled sim_hash and the new sim_hash have in common.
        # If that number is greater than 31, then return True
        if len([bit for bit in range(32) if hashed_page_value[bit] == hashed_value_bits[bit]]) > 31: 
            print(f"Not Includeing this because it is too similar to {hashed_page_url}")
            return True
    
    # If new url is not similar to old ones then add url and its sim_hash to the hashed pages dictionary.
    hashed_pages[url] = hashed_value

    # Save updated dictionary to file.
    save_dict_to_pickle(hashed_pages, filepath)
    return False


def search(query: str) -> list[tuple[int, int]]:
    """Searches for documents that match query."""
    # Tokenize and stem query
    tokenized_query = indexer.tokenize(query)
    if len(tokenized_query) > 3:
        nGrams = NGramTokenizer(query)
    else:
        nGrams = None
    # Remove stop words
    query_tokens = [token for token in tokenized_query if token not in stop_words]
    print(f"Stemmed to: {query_tokens}")
    # Compute word frequencies of query
    frequencies, nGramFrequencies = computeQueryFrequencies(query_tokens, nGrams)
    # Get postings of all tokens
    postings = getPostings(query_tokens)
    nGramPostings = getNGramPostings(nGrams, postings)
    # Calculate tf-idf of query
    query_normalized_weights, query_nGram_normalized_weights = calculateQueryNormalizedTfIdf(query_tokens, postings, nGramPostings, frequencies, nGrams, nGramFrequencies)
    # Calculate tf-wt of documents
    document_normalized_weights = calculateDocumentNormalizedTfWt({token: posting for (token, posting) in postings.items() if token in query_normalized_weights}, {ngram:posting for (ngram, posting) in nGramPostings.items() if ngram in query_nGram_normalized_weights})
    # Calcualte cosine similarity values of each document
    document_scores = defaultdict(int)
    for document, token_normalized_scores in document_normalized_weights.items():
        for token, score in token_normalized_scores.items():
            if isinstance(token, str):
                document_scores[document] += score * query_normalized_weights[token]
            else:
                document_scores[document] += score * query_nGram_normalized_weights[token]
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
    location_index_0, location_index_1, location_index_2, location_index_3, location_index_4, page_rank_index, sim_hash_index = start_up()
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
        # for document_info in urls:
            # print(document_info)
        print(f"\nFound {len(sorted_postings)} results in {end_time - start_time} seconds\n")
