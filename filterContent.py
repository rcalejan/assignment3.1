"""File holds definitions for functions that filter data for indexing."""

###################################################################
#                           Imports                               #
###################################################################
from porter2stemmer import Porter2Stemmer
import re


###################################################################
#                Global Variabls and Constants                    #
###################################################################
stemmer = Porter2Stemmer()


###################################################################
#                          Functions                              #
###################################################################
def countFrequencyAndPosition(tokens: list[str]) -> dict[str, int]:
    """Compute token frequency and its positions in document."""
    frequencies = dict()
    for position, word in enumerate(tokens):
        if word not in frequencies:
            frequencies[word] = [1, [position]]
        else:
            frequencies[word][0] += 1
            frequencies[word][1].append(position)
    return frequencies


def singleTokenize(file_content) -> list[str]:
    """Tokenizes single tokens. Tokens are sequencies 
    of numbers, characters, or underscores."""
    file_content = file_content.lower()
    pattern = re.compile(r'\w+')
    tokens = pattern.findall(file_content)
    porterized = map(stemmer.stem, tokens)
    return list(porterized)


def nGramTokenize(single_tokens: list[str]) -> list[str]:
    """Tokenizes bigrams and trigrams."""
    n_grams = []
    for i in range(len(single_tokens)-2):
        bigram = single_tokens[i] + ' ' + single_tokens[i+1]
        trigram = bigram + ' ' + single_tokens[i+2]
        n_grams.append(bigram)
        n_grams.append(trigram)
    n_grams.append(single_tokens[len(single_tokens)-2] + ' ' + single_tokens[len(single_tokens)-1])
    return n_grams


def tokenize(file_content) -> list[str]:
    """Tokenizes file_content and returns tokens in list including bigrams and trigrams."""
    tokens = singleTokenize(file_content)
    if len(tokens) < 2:
        return tokens
    if len(tokens) < 3:
        tokens.append(tokens[0] + ' ' + tokens[1])
        return tokens
    tokens.extend(nGramTokenize(tokens))
    return tokens


def queryTokenize(file_content) -> list[str]:
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