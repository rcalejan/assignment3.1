'''
This is where the ranking algorithm goes.
Look into tf-idf ranking
'''
###################################################################
#                           Imports                               #
###################################################################
from graph import Graph, Node, runHits
from collections import defaultdict
import pickle
import os
from glob import glob
import json
import orjson
import time
from jsonhandling import load_index, load_page_rank_index, load_token_idf_index
from filterContent import queryTokenize
from ngram import getNGramPostings, getPostingsForNGram, NGramTokenizer
from tfIdf import calculateDocumentNormalizedTfWt, calculateQueryNormalizedTfIdf
import re
from flask import Flask, render_template, request
import openai
###################################################################
#                Global Variabls and Constants                    #
###################################################################

openai.api_key = os.getenv("OPEN_AI_KEY")
PAGE_RANK_FACTOR = 1000
HITS_AUTHORITY_FACTOR = 100
HITS_HUB_FACTOR = 100
DOCUMENT_PATHS = [file for dir in os.walk('developer/DEV') for file in glob(os.path.join(dir[0], '*.json'))]
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
app = Flask(__name__)


###################################################################
#                          Functions                              #
###################################################################

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
    page_rank_index = load_page_rank_index()

    # Load token_idf_index
    token_idf_index = load_token_idf_index()

    # Load webGraph
    print("Loading Web Graph...")
    with open(f'./webGraph/webGraph.pickle', 'rb') as pickle_file:
        web_graph = Graph()
        # Convert WebGraph to Node version so we can do math on it for HITS Algorithm
        web_graph.nodes = pickle.load(pickle_file)
        web_graph.convertGraphForLoading()
    print("\tFinished")
    
    return location_index_0, location_index_1, location_index_2, location_index_3, location_index_4, page_rank_index, token_idf_index, web_graph


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
            for new_posting in findPostings(token, championList=True): # Get champion list for tokens
                all_postings[token].append(new_posting) # Add postings to return dictionary
    return all_postings


def findPostings(token: str, championList = True) -> list:
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
    
    # If Champion List requested only get top 100 postings
    if championList:
        with open(f'./finalIndex/index{index_num}.json', 'r') as final_index:
            final_index.seek(start_position)    # Jump to file start position
            posting_data = final_index.read(hundred_end_postiion-start_position)    # Read object string up until end of top 100 postings
        postings = orjson.loads(posting_data)    # Load json string into dictionary
    
    # If all postings are neccessary
    else:
        with open(f'./finalIndex/index{index_num}.json', 'r') as final_index:
            final_index.seek(start_position)    # Jump to file start position
            posting_data = final_index.read(hundred_end_postiion-start_position)    # Read object string up until end of top 100 postings
            postings = orjson.loads(posting_data)    # Load json string into dictionary
            final_index.seek(hundred_end_postiion)
            posting_data = final_index.read(end_position-hundred_end_postiion)    # Read object string from end of top 100 postings to end of postings
            postings.extend(orjson.loads(posting_data)) # Combine two lists
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


def search(query: str) -> list[tuple[int, int]]:
    """Searches for documents that match query."""

    # Tokenize and stem query
    tokenized_query, num_single_tokens = queryTokenize(query)

    # Only use top 10 tokens by idf for efficient retrieval
    if num_single_tokens > 4:
        tokenized_query = sorted([token for token in tokenized_query if isinstance(token, str) and token in token_idf_index], key= lambda token: token_idf_index[token], reverse=True)[:10]
    print(tokenized_query)

    # If the length of the query is greater than six then don't use NGrams for indexing because the current tokens perform better
    if num_single_tokens and num_single_tokens < 6:
        nGrams = NGramTokenizer(query)
    else:
        nGrams = None

    # Remove stop words
    query_tokens = [token for token in tokenized_query if token not in stop_words]
    
    # Compute word frequencies of query
    frequencies, nGramFrequencies = computeQueryFrequencies(query_tokens, nGrams)
    
    # Get postings of all tokens
    postings = getPostings(query_tokens)
    nGramPostings = getNGramPostings(nGrams, postings)

    # Calculate tf-idf of query
    query_normalized_weights, query_nGram_normalized_weights = calculateQueryNormalizedTfIdf(query_tokens, postings, nGramPostings, frequencies, nGrams, nGramFrequencies, token_idf_index)
    
    # Calculate tf-wt of documents
    document_normalized_weights = calculateDocumentNormalizedTfWt({token: posting for (token, posting) in postings.items() if token in query_normalized_weights}, {ngram:posting for (ngram, posting) in nGramPostings.items() if ngram in query_nGram_normalized_weights})

    # Calcualte cosine similarity values of each document
    document_scores = defaultdict(int)
    for document, token_normalized_scores in document_normalized_weights.items():
        document_scores[document] += page_rank_index[str(document)] * PAGE_RANK_FACTOR
        for token, score in token_normalized_scores.items():
            if isinstance(token, str):
                document_scores[document] += score * query_normalized_weights[token]
            else:
                document_scores[document] += score * query_nGram_normalized_weights[token]
    
    # Sort documents by score
    sorted_list_of_documents = sorted(document_scores.items(), key = lambda document: document[1], reverse=True)

    # Def Run Hits Algorithm on top 500 or fewer Documents Found
    hits_graph_input = [document[0] for document in sorted_list_of_documents[:100]]
    hits_graph = runHits(hits_graph_input, web_graph)

    # Add hits scores to document scores
    for document in hits_graph.nodes.keys():
        document_scores[document] += hits_graph.nodes[document].authority * HITS_AUTHORITY_FACTOR + hits_graph.nodes[document].hub * HITS_HUB_FACTOR

    # Sort Documents again
    sorted_list_of_documents = sorted(document_scores.items(), key = lambda document: document[1], reverse=True)
    return sorted_list_of_documents


def getTopKUrls(documents: list[tuple[int, int]], k: int) -> list[tuple[int, str]]:
    """Return urls of the top k documents sorted by score."""
    urls = []
    for index, (documentid, _) in enumerate(documents):
        if index == k:
            break
        doc_path = DOCUMENT_PATHS[documentid]
        with open(doc_path, 'r') as file:
            data = json.load(file)
            urls.append(data['url'])
    return urls


@app.route('/')
def mainPage():
    """Return main index.html"""
    return render_template('index.html')


@app.route('/api')
def searchQuery():
    """Return results of query"""
    print("SDSDFSFD")
    query = request.args.get('query')
    # Keep track of when index started
    start_time = time.time()

    print(f"Getting results for query: {query}")
    # Main Search
    sorted_postings = search(query)  

    # Get Top 10 Url's from Search
    urls = getTopKUrls(sorted_postings, 10) 

    # Keep track of when seach finished
    end_time = time.time()
    return json.dumps((len(sorted_postings), end_time-start_time, urls))
    

def summarizePage(page):
    """Summarizes each result using the OpenAI API"""
    prompt = f'Can you summarize this page in less than 50 words: {page}'
    completion = openai.Completion.create(
        engine='gpt-3.5-turbo',
        prompt=prompt,
        max_tokens=100,
    )
    return completion.choices[0].text

###################################################################
#                             Main                                #
###################################################################

if __name__ == '__main__':
    # Load location indexes into memory
    print("Starting Up")
    location_index_0, location_index_1, location_index_2, location_index_3, location_index_4, page_rank_index, token_idf_index, web_graph = start_up()
    print("Done")

    # Run Flask App
    app.run(debug=True)

        
        