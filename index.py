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
from zlib import crc32
import datetime
from urllib.parse import urljoin
from glob import glob
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
sim_hashes = {}
DOCUMENT_PATHS = [file for dir in os.walk('developer/DEV') for file in glob(os.path.join(dir[0], '*.json'))]


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
    

class Graph:
    """Graph of entire courpus."""
    def __init__(self) -> None:
        self.nodes = {}


    def addNode(self, hashed_url:int, node:object):
        self.nodes[hashed_url] = node


    def convertUrlsToNodes(self) -> None:
        for node in self.nodes.values():
            for child in node.children_urls:
                if child in self.nodes:
                    self.nodes[child].parents.add(node)
        for node in self.nodes.values():
            for parent in node.parents:
                parent.children.add(node)


    def convertGraphForSaving(self) -> None:
        self.nodes = {node.docID: node for node in self.nodes.values()}
        for node in self.nodes.values():
            for child in [node for node in node.children]:
                node.children.remove(child)
                node.children.add(child.docID)
            for parent in [node for node in node.parents]:
                node.parents.remove(parent)
                node.parents.add(parent.docID)

    
    def convertGraphForLoading(self) -> None:
        for node in self.nodes.values():
            for docId in [node for node in node.children]:
                node.children.remove(docId)
                node.children.add(self.nodes[docId])
            for docId in [node for node in node.parents]:
                node.parents.remove(docId)
                node.parents.add(self.nodes[docId])


    def runHits(self):
        for node in self.nodes.values():
            node.updateAuthority()
        for node in self.nodes.values():
            node.updateHub()
        self.normalize_nodes()


    def runPageRank(self, d):
        for node in self.nodes.values():
            node.updatePageRank(d, len(self.nodes))


    def normalize_nodes(self) -> None:
        authority_sum = sum(node.authority for node in self.nodes.values())
        hub_sum = sum(node.hub for node in self.nodes.values())
        for node in self.nodes.values():
            if authority_sum:
                node.authority /= authority_sum
            if hub_sum:
                node.hub /= hub_sum

    
    def save(self):
        self.convertGraphForSaving()
        with open(f'webGraph/webGraph.pickle', 'wb') as web_graph_file:
            pickle.dump(self.nodes, web_graph_file, pickle.HIGHEST_PROTOCOL)
        print("\tSaved Web Graph")
    


class Node:
    """Node representation of document in corupus."""
    def __init__(self, hashed_url: int, docID: int) -> None:
        self.url = hashed_url
        self.children_urls = set()
        self.children = set()
        self.parents = set()
        self.authority = 1.0
        self.hub = 1.0
        self.page_rank = 1.0
        self.docID = docID


    def updateAuthority(self):
        self.authority = sum(node.hub for node in self.parents)


    def updateHub(self):
        self.hub = sum(node.authority for node in self.children)


    def updatePageRank(self, damping_factor, n):
        pagerank_sum = sum((node.page_rank / len(node.children)) for node in self.parents)
        random_walk = damping_factor / n
        self.page_rank = random_walk + (1 - damping_factor) * pagerank_sum


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


def simHash(tokens: dict[str:int]) -> int:
    """Returns simhash value of tokenized words.
       Uses crc32."""
    
    # Initialize 32 bit vector
    vector = [0 for i in range(32)]
    num_bits = 32

    # Update vector for each word
    for key, value in tokens.items(): # Iterate through words and their counts
        value = value[0]
        hashed_value = crc32(bytes(key, 'utf-8')) # Compute 32 bit hash value for word
        bits = [(hashed_value >> bit) & 1 for bit in range(num_bits - 1, -1, -1)] # Convert integer hash value to list of binary digits
        for index, bit in enumerate(bits): # Iterate through bianry digits
            if bit == 1:
                vector[index] += value # If binary digit is 1 then add word count to vector position of binary digit
            elif bit == 0:
                vector[index] -= value # If binary digit is 1 then subtract word count to vector position of binary digit
    
    binary_list = [1 if value >=0 else 0 for value in vector] # Convert vector into binary.  1 if vector[i] is positive. 0 Otherwise.
    integer = int("".join(str(x) for x in binary_list), 2) # Convert binary to integer
    return integer # Return integer representation of hashed value


def isSimilar(frequencies: dict[str:int]) -> bool:
    """Checks if file tokens are similar to a previously crawled page using sim_hash."""

    # Calcualte sim_hash value of the tokens using custom sim_hash function above
    hashed_value = simHash(frequencies)
    
    # Iterate through previously crawled documents and their sim_hash values
    for docId, value in sim_hashes.items():
        
        # Count how many bits the previously crawled sim_hash and the new sim_hash have in common.
        # If that number is greater than 31, then return True
        if bin(hashed_value ^ value).count("1") > 31: 
            print(f"Not Includeing doc {current_id} because it is too similar to doc {docId}")
            print(DOCUMENT_PATHS[current_id])
            print(DOCUMENT_PATHS[docId])
            return True
    
    # If new document is not similar to old ones then add document and its sim_hash to the hashed pages dictionary.
    sim_hashes[current_id] = hashed_value
    return False


def saveHashes(index: dict[str, list]) -> None:
    """Dump hash dictionary to json file."""
    with open(f'./docHashes/hashIndex.json', 'w') as json_file:
        print(f"{datetime.datetime.now()} Started Saving Hash Index")
        json.dump(index, json_file)
        print(f"{datetime.datetime.now()} Finished Saving Hash Index")


def savePageRanks(index: dict[str, list]) -> None:
    """Dump hash dictionary to json file."""
    with open(f'./pageRanks/pageRankIndex.json', 'w') as json_file:
        print(f"{datetime.datetime.now()} Started Saving Page Ranks")
        json.dump(index, json_file)
        print(f"{datetime.datetime.now()} Finished Saving Page Ranks")


def indexFile(file: str):
    """Reads content of file, tokenizes it, creates posting, and adds posting to main index."""
    with open(file) as json_file:
        file_data = json.load(json_file)
        soup = BeautifulSoup(file_data['content'], 'lxml')

        tokens = tokenize(soup.get_text())
        frequencies = countFrequencyAndPosition(tokens)

        if isSimilar(frequencies):
            return
        
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

        node = Node(hash(file_data['url']), current_id)
        for link in soup.find_all('a'): # Iterate thorugh a tags in soup
            if link and link.get('href'): # Get url from link
                node.children_urls.add(hash(urljoin(file_data['url'], link.get('href')))) # Convert relative url to absolute url and append it to links list
        webGraph.addNode(node.url, node)

###################################################################
#                             Main                                #
###################################################################

if __name__=='__main__':
    webGraph = Graph()
    for index, (subdir, dirs, files) in enumerate(os.walk('./developer/DEV')):
        print(subdir)
        for file in tqdm(files):
            indexFile(subdir + '/' + file)
            if(current_id % 1000 == 0 and current_id != 0):
                dumpToPickle(main_index, file_number)
                main_index = defaultdict(list)
                file_number += 1
            current_id += 1
    webGraph.convertUrlsToNodes()
    for i in range(5):
        print(f"{datetime.datetime.now()} Started PageRank Iteration {i} of 5")
        webGraph.runPageRank(0.85)
        print(f"{datetime.datetime.now()} Finished Iteration")
    print(current_id)
    dumpToPickle(main_index, file_number)
    saveHashes(sim_hashes)
    savePageRanks({node.docID:node.page_rank for node in webGraph.nodes.values()})
    webGraph.save()
    merge()
    createFinalIndex()
