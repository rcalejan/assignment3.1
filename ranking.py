from index import singleTokenize, nGramTokenize, Graph, Node
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
PAGE_RANK_FACTOR = 1000
HITS_AUTHORITY_FACTOR = 100
HITS_HUB_FACTOR = 100
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


def start_up() -> tuple:
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

    # Load token_idf_index
    print("Loading tokenIdfIndex Index...")
    with open(f'./tokenIdfIndex/tokenIdf.pickle', 'rb') as pickle_file:
        token_idf_index = pickle.load(pickle_file)
    print("\tFinished")

    # Load webGraph
    print("Loading Web Graph...")
    with open(f'./webGraph/webGraph.pickle', 'rb') as pickle_file:
        web_graph = Graph()
        web_graph.nodes = pickle.load(pickle_file)
        web_graph.convertGraphForLoading()
    print("\tFinished")
    
    return location_index_0, location_index_1, location_index_2, location_index_3, location_index_4, page_rank_index, sim_hash_index, token_idf_index, web_graph


def askUser()->str:
    """Prompts User for query and returns input string"""
    print("==========  Welcome to GHoogle!  ==========\n")
    query = input("What do you want to know? ('quit' to exit): ")
    return query


def tokenize(file_content) -> list[str]:
    """Tokenizes file_content and returns tokens in list including bigrams and trigrams."""
    tokens = singleTokenize(file_content)
    num_single_tokens = len(tokens)
    if num_single_tokens < 2:
        return tokens, num_single_tokens
    if num_single_tokens < 3:
        tokens.append(tokens[0] + ' ' + tokens[1])
        return tokens, num_single_tokens
    tokens.extend(nGramTokenize(tokens))
    return tokens, num_single_tokens


def getPostings(tokens: list) -> dict[list[list]]:
    """Gets all postings for every token and n gram."""
    all_postings = defaultdict(list)  # Initialize Dictionary
    for token in set(tokens):
        if token not in stop_words:  # If token is a stop word then don't look for it's posting
            for new_posting in findPostings(token):
                all_postings[token].append(new_posting)
    return all_postings


def getCombinedPostings(postings, distance, first_word, second_word):
    """Returns postings that contain both words."""
    first_word_postings = postings[first_word]
    second_word_postings=  postings[second_word]
    first_pos = 0
    second_pos = 0
    first_id = first_word_postings[first_pos][0]
    second_id = second_word_postings[second_pos][0]
    max_id = min(first_word_postings[-1][0], second_word_postings[-1][0])
    n_gram_postings = []

    while first_id < max_id and second_id < max_id:
        if second_id < first_id:
            second_pos += 1
            second_id = second_word_postings[second_pos][0]
        elif first_id < second_id:
            first_pos += 1
            first_id = first_word_postings[first_pos][0]
        else:
            for first_word_position in first_word_postings[first_pos][2]:
                second_word_position = first_word_position + distance
                if second_word_position in second_word_postings[second_pos][2]:
                    n_gram_postings.append((first_word_postings[first_pos], second_word_postings[second_pos]))
                    break
            first_pos += 1
            first_id = first_word_postings[first_pos][0]
            second_pos += 1
            second_id = second_word_postings[second_pos][0]
    return n_gram_postings


def getNGramPostings(nGrams, postings):
    """Get postings for each n gram"""
    nGramPostings = {}
    if nGrams:
        for nGram in nGrams:
            distance, first_word, second_word = nGram
            if first_word in postings and second_word in postings:
                nGramPostings[nGram] = getCombinedPostings(postings, distance, first_word, second_word)
    return nGramPostings
        

def findPostings(token: str, first_hundred = True) -> list:
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
        start_position, hundred_end_postiion, end_position = index_to_use[token]  # Get start and end position of json list in file
    except KeyError:
        return []    # if token wasn't found return empty list
    if first_hundred:
        with open(f'./finalIndex/index{index_num}.json', 'r') as final_index:
            final_index.seek(start_position)    # Jump to file start position
            posting_data = final_index.read(hundred_end_postiion-start_position)    # Read object string
        postings = orjson.loads(posting_data)    # Load json string into dictionary
    else:
        with open(f'./finalIndex/index{index_num}.json', 'r') as final_index:
            final_index.seek(start_position)    # Jump to file start position
            posting_data = final_index.read(hundred_end_postiion-start_position)    # Read object string
            postings = orjson.loads(posting_data)    # Load json string into dictionary
            final_index.seek(hundred_end_postiion+1)
            posting_data = final_index.read(end_position-hundred_end_postiion+1)    # Read object string
            postings.extend(orjson.loads(posting_data))
    return postings


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
        query_idf[token] = token_idf_index[token]

    # Only use top 20 idf's to search (for efficiency)
    if len(query_idf) > 20:
        for token, _ in sorted(query_idf.items(), key= lambda x: x[1], reverse=True)[10:]:
            query_idf.pop(token)

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


def runHits(documents):
    start_time = time.time()
    hits_graph = Graph()
    for document in documents:
        hits_graph.nodes[document] = web_graph.nodes[document]
    for node in [node for node in hits_graph.nodes.values()]:
        for parent in sorted([parent for parent in node.parents], key = lambda node: node.page_rank, reverse=True)[:10]:
            hits_graph.nodes[parent.docID] = parent
    end_time = time.time()
    print(f"JOE Took {end_time - start_time} seconds")
    start_time = time.time()
    print(f"Increased to {len(hits_graph.nodes)} nodes")
    for i in range(5):
        hits_graph.runHits()
    end_time = time.time()
    print(f"McFlow Took {end_time - start_time} seconds")
    return hits_graph


def search(query: str) -> list[tuple[int, int]]:
    """Searches for documents that match query."""

    # Tokenize and stem query
    start_time = time.time()
    tokenized_query, num_single_tokens = tokenize(query)
    if num_single_tokens > 4:
        tokenized_query = sorted([token for token in tokenized_query if isinstance(token, str) and token in token_idf_index], key= lambda token: token_idf_index[token], reverse=True)[:10]
    print(tokenized_query)
    if num_single_tokens and num_single_tokens < 6:
        nGrams = NGramTokenizer(query)
    else:
        nGrams = None
    end_time = time.time()
    print(f"1 Took {end_time - start_time} seconds")
    start_time = time.time()
    
    # Remove stop words
    query_tokens = [token for token in tokenized_query if token not in stop_words]
    print(f"Stemmed to: {query_tokens}")
    end_time = time.time()
    print(f"2 Took {end_time - start_time} seconds")
    start_time = time.time()
    print(nGrams)
    
    # Compute word frequencies of query
    frequencies, nGramFrequencies = computeQueryFrequencies(query_tokens, nGrams)
    end_time = time.time()
    print(f"3 Took {end_time - start_time} seconds")
    start_time = time.time()
    
    # Get postings of all tokens
    postings = getPostings(query_tokens)
    end_time = time.time()
    print(f"4 Took {end_time - start_time} seconds")
    start_time = time.time()
    nGramPostings = getNGramPostings(nGrams, postings)
    end_time = time.time()
    print(f"5 Took {end_time - start_time} seconds")
    start_time = time.time()
    
    # Calculate tf-idf of query
    query_normalized_weights, query_nGram_normalized_weights = calculateQueryNormalizedTfIdf(query_tokens, postings, nGramPostings, frequencies, nGrams, nGramFrequencies)
    print(query_normalized_weights, query_nGram_normalized_weights)
    end_time = time.time()
    print(f"6 Took {end_time - start_time} seconds")
    start_time = time.time()
    
    # Calculate tf-wt of documents
    document_normalized_weights = calculateDocumentNormalizedTfWt({token: posting for (token, posting) in postings.items() if token in query_normalized_weights}, {ngram:posting for (ngram, posting) in nGramPostings.items() if ngram in query_nGram_normalized_weights})
    end_time = time.time()
    print(f"7 Took {end_time - start_time} seconds")
    print(f"222 Took {end_time - start_time} seconds")
    start_time = time.time()
    # Calcualte cosine similarity values of each document
    document_scores = defaultdict(int)
    for document, token_normalized_scores in document_normalized_weights.items():
        document_scores[document] += page_rank_index[str(document)] * PAGE_RANK_FACTOR
        for token, score in token_normalized_scores.items():
            if isinstance(token, str):
                document_scores[document] += score * query_normalized_weights[token]
            else:
                document_scores[document] += score * query_nGram_normalized_weights[token]

    end_time = time.time()
    print(f"8 Took {end_time - start_time} seconds")
    start_time = time.time()
    
    # Sort documents by score
    sorted_list_of_documents = sorted(document_scores.items(), key = lambda document: document[1], reverse=True)
    
    end_time = time.time()
    start_time = time.time()
    # Def Run Hits Algorithm on Documents Found
    hits_graph_input = [document[0] for document in sorted_list_of_documents[:100]]
    print(f"running hits on length {len(hits_graph_input)}")
    hits_graph = runHits(hits_graph_input)
    max_authority = 0
    max_hub = 0
    for document in hits_graph.nodes.keys():
        document_scores[document] += hits_graph.nodes[document].authority * HITS_AUTHORITY_FACTOR + hits_graph.nodes[document].hub * HITS_HUB_FACTOR
        if hits_graph.nodes[document].authority > max_authority:
            max_authority = hits_graph.nodes[document].authority
        if hits_graph.nodes[document].hub > max_hub:
            max_hub = hits_graph.nodes[document].hub
    print(f"Max Hub: {max_hub}")
    print(f"Max Authority: {max_authority}")
    end_time = time.time()
    print(f"9 Took {end_time - start_time} seconds")
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
    location_index_0, location_index_1, location_index_2, location_index_3, location_index_4, page_rank_index, sim_hash_index, token_idf_index, web_graph = start_up()
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
            print(document_info['url'])
        print(f"\nFound {len(sorted_postings)} results in {end_time - start_time} seconds\n")
