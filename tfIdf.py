"""File hodls function definitions for tf and idf computing."""
###################################################################
#                           Imports                               #
###################################################################
from math import log, sqrt
from collections import defaultdict


###################################################################
#                Global Variabls and Constants                    #
###################################################################

NUMBER_OF_FILES = 55393
TITLE_VALUE = 20
BOLD_VALUE = 5
HEADER_VALUE = 10


###################################################################
#                          Functions                              #
###################################################################

def calculateQueryNormalizedTfIdf(query_tokens: list[str], postings: list[list], nGramPostings: dict[tuple, list], frequencies: dict[str, int], nGrams: list[tuple[int, str, str]], nGramFrequencies: dict[tuple, int], token_idf_index) -> dict[str, int]:
    """Calculates normalized tf-idf for query tokens."""

    # Calculate tf-wt
    query_tf_wt = {}
    tokens = set(query_token for query_token in query_tokens if query_token in postings and postings[query_token])
    for token in tokens:
        query_tf_wt[token] = 1 + log(frequencies[token], 10)
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
    query_normalization_length = sqrt(sum([i**2 for i in query_weights.values()]))
    for token, weight in query_weights.items():
        query_normalized_scores[token] = weight/query_normalization_length

    # Calculate tf-wt for nGrams
    if nGrams:
        query_tf_wt = {}
        nGrams = [nGram for nGram, posting_list in nGramPostings.items() if posting_list]
        for nGram in nGrams:
            query_tf_wt[nGram] = 1 + log(nGramFrequencies[nGram], 10)

        # Calcualte idf for nGrams
        query_idf = {}
        for nGram in nGrams:
            query_idf[nGram] = log(NUMBER_OF_FILES/len(nGramPostings[nGram]), 10)

        # Claculate tf-idf for nGrams
        query_weights = {}
        for token in query_idf.keys():
            query_weights[token] = query_tf_wt[token] * query_idf[token]

        # Normalize tf-idf for nGrams
        query_nGram_normalized_scores = {}
        query_normalization_length = sqrt(sum([i**2 for i in query_weights.values()]))
        for token, weight in query_weights.items():
            query_nGram_normalized_scores[token] = weight/query_normalization_length
    else:
        query_nGram_normalized_scores = {}

    return query_normalized_scores, query_nGram_normalized_scores


def calculateDocumentNormalizedTfWt(postings: dict[str, list], nGramPostings: dict[tuple, list]) -> dict:
    """Calcualte normalized tf scores for each document."""
    document_term_scores = defaultdict(dict)
    for token, posting in postings.items():
        for docID, freq, _, title, bold, header in posting:
            # Calcualte tf-wt and add scores for if token was a title, bold, or header
            document_term_scores[docID][token] = 1 + log(freq, 10) + title * TITLE_VALUE + bold * BOLD_VALUE + header * HEADER_VALUE
        # Normalize Values
        document_term_scores[docID][token] = document_term_scores[docID][token]/sum(document_term_scores[docID].values())
    
    # Same but for N Grams
    for nGram, first_second_postings in nGramPostings.items():
        for posting in first_second_postings:
            for docID, freq, _, title, bold, header in posting:
                # Calcualte tf-wt and add scores for if nGram was a title, bold, or header
                document_term_scores[docID][nGram] = 1 + log(freq, 10) + title * TITLE_VALUE + bold * BOLD_VALUE + header * HEADER_VALUE
            # Normalize Values
            document_term_scores[docID][nGram] = document_term_scores[docID][nGram]/sum(document_term_scores[docID].values())
    return document_term_scores