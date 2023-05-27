"""This file is responsible for indexing the entire corpus.  It first indexes the corpus in batches saving
each batch to a pickle dump under /dumps/.  Then it iterates through the dumps and creates 5 master indexes
stored under /masterIndex/ that hold the combined information.  These master indexes are split by first 
characters of their tokens.  Finally, the master indexes are read are written to 5 final indexes stored under
/finalIndex/.  During this process 5 location indexes are created and stored under /indexIndex/ that store the 
start and stop positions of the postings assigned to each token in the final Index."""


from bs4 import BeautifulSoup
from collections import defaultdict
import os
import json
import re
from porter2stemmer import Porter2Stemmer
from jsonhandling import dumpToPickle, merge, createFinalIndex
from tqdm import tqdm
import time
import sys
import pickle

###################################################################
#                          Constants                              #
###################################################################
current_id = 0
main_index = defaultdict(list)
stemmer = Porter2Stemmer()
file_number = 1

###################################################################
#                           Classes                               #
###################################################################
class Posting:
    """Holds posting information"""
    def __init__(self, docID: int, freq: int, posList: list[int], title: bool, bold: bool, header: bool):
        self.docID = docID      # int: Document Integer ID
        self.freq = freq        # int: frequency of token in Doc
        self.posList = posList  # list[int]: position of the token in the doc 
        self.title = title      # bool: does this token appear in the title in Doc
        self.bold = bold        # bool: is this token in a bold tag in the Doc
        self.header = header    # bool: does this token appear in an h1, h2, or h3 tag in the Doc
    

    def toList(self) -> list:
        """Returns posting as list for serialization."""
        return [self.docID, self.freq, self.posList, self.title, self.bold, self.header]


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


def indexFile(file):
    """Reads content of file, tokenizes it, creates posting, and adds posting to main index."""
    with open(file) as json_file:
        file_data = json.load(json_file)
        soup = BeautifulSoup(file_data['content'], 'lxml')
        headers = set()
        for header1 in soup.find_all('h1'):
            for header in tokenize(header1.text):
                headers.add(header)
        for header2 in soup.find_all('h2'):
            for header in tokenize(header2.text):
                headers.add(header)
        for header3 in soup.find_all('h3'):
            for header in tokenize(header3.text):
                headers.add(header)

        bolded = set()
        for bold_tags in soup.find_all('strong'):
            for bolded_word in tokenize(bold_tags.text):
                bolded.add(bolded_word)
        for bold_tags in soup.find_all('b'):
            for bolded_word in tokenize(bold_tags.text):
                bolded.add(bolded_word)

        titles = set()
        for title_tags in soup.find_all('title'):
            for title in tokenize(title_tags.text):
                titles.add(title)

        tokens = tokenize(soup.get_text())
        frequencies = countFrequencyAndPosition(tokens)
        for token, (freq, positions) in frequencies.items():
            isTitle = False
            isBold = False
            isHeader = False
            if token in titles:
                isTitle = True
            if token in bolded:
                isBold = True
            if token in headers:
                isHeader = True
            main_index[token].append((current_id, freq, positions, isTitle, isBold, isHeader))


###################################################################
#                             Main                                #
###################################################################

if __name__=='__main__':
    for subdir, dirs, files in os.walk('./developer/DEV'):
        print(subdir)
        for file in tqdm(files):
            indexFile(subdir + '/' + file)
            if(current_id % 1000 == 0 and current_id != 0):
                dumpToPickle(main_index, file_number)
                main_index = defaultdict(list)
                file_number += 1
            current_id += 1
    print(current_id)
    dumpToPickle(main_index, file_number)
    merge()
    createFinalIndex()
