"""This file is responsible for indexing the entire corpus.  It first indexes the corpus in batches saving
each batch to a pickle dump under /dumps/.  Then it iterates through the dumps and creates 5 master indexes
stored under /masterIndex/ that hold the combined information.  These master indexes are split by first 
characters of their tokens.  Finally, the master indexes are read are written to 5 final indexes stored under
/finalIndex/.  During this process 5 location indexes are created and stored under /indexIndex/ that store the 
start and stop positions of the postings assigned to each token in the final Index."""

###################################################################
#                           Imports                               #
###################################################################
from bs4 import BeautifulSoup
from collections import defaultdict
import os
import json
from tqdm import tqdm
import datetime
import sys
from urllib.parse import urljoin
from graph import Graph, Node
from jsonhandling import dumpToPickle, savePageRanks, saveHashes
from filterContent import tokenize, countFrequencyAndPosition
from hash import isSimilar
from directorySetUp import setUpDirectory
from merge import merge, createFinalIndex

###################################################################
#                Global Variabls and Constants                    #
###################################################################
current_id = 0
main_index = defaultdict(list)
file_number = 1
sim_hashes = {}

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


def indexFile(file: str):
    """Reads content of file, checks for similarity, tokenizes it, creates 
        posting, adds posting to main index, creates Node, and adds Node to 
        Web Graph."""
    with open(file) as json_file:

        # Load Data
        file_data = json.load(json_file)
        soup = BeautifulSoup(file_data['content'], 'lxml')

        # Tokenize and Count Frequency and Position
        tokens = tokenize(soup.get_text())
        frequencies = countFrequencyAndPosition(tokens)

        # Check if document is too similar to previously indexed document
        if isSimilar(frequencies, sim_hashes, current_id):
            return
        
        # Mark token as header
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

        # Mark token as bold
        bolded = set()
        for bold_tags in soup.find_all('strong'):
            for bolded_word in tokenize(bold_tags.text):
                bolded.add(bolded_word)
        for bold_tags in soup.find_all('b'):
            for bolded_word in tokenize(bold_tags.text):
                bolded.add(bolded_word)

        # Mark token as title
        titles = set()
        for title_tags in soup.find_all('title'):
            for title in tokenize(title_tags.text):
                titles.add(title)

        # Create and add postings for each token
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

        # Create Node
        node = Node(file_data['url'], current_id)

        # Add Links to Node children
        for link in soup.find_all('a'): # Iterate thorugh a tags in soup
            if link and link.get('href'): # Get url from link
                node.children_urls.add(urljoin(file_data['url'], link.get('href'))) # Convert relative url to absolute url and append it to links list

        # Add Node to graph
        webGraph.addNode(node.url, node)

###################################################################
#                             Main                                #
###################################################################

if __name__=='__main__':
    """Main code that indexes developer folder."""
    print(f"{datetime.datetime.now()} Checking Directory")
    if not setUpDirectory():
        sys.exit()
    print(f"{datetime.datetime.now()} Directory Check Complete")


    print(f"{datetime.datetime.now()} Starting Indexing")

    # Create Web Graph
    webGraph = Graph()

    # Iterate through files in corpus
    for index, (subdir, dirs, files) in enumerate(os.walk('./developer/DEV')):
        print(f"{datetime.datetime.now()} Indexing Sub Directory: {subdir}")
        for file in tqdm(files):
            # Index file
            indexFile(subdir + '/' + file)

            # Dump File if it has 1000 tokens
            if(current_id % 1000 == 0 and current_id != 0):
                dumpToPickle(main_index, file_number)
                main_index = defaultdict(list)
                file_number += 1
            
            # Increment DocId
            current_id += 1
    print(f"Indexed {current_id} documents")

    # Dump remaining information to file
    dumpToPickle(main_index, file_number)

    # Save Hashed Values to File
    saveHashes(sim_hashes)

    # Convert Web Graph to Node version for running Page Rank Algorithm
    webGraph.convertUrlsToNodes()

    # Run 5 iterations of Page Rank Algorithm
    for i in range(5):
        print(f"{datetime.datetime.now()} Started PageRank Iteration {i} of 5")
        webGraph.runPageRank(0.85)
        print(f"{datetime.datetime.now()} Finished Iteration")

    # Save Page Rank values of each DocID to file
    savePageRanks({node.docID:node.page_rank for node in webGraph.nodes.values()})

    # Save Web Graph to file
    webGraph.save()

    # Merge All Dumps into 5 seperate combined indexes organized by tokens
    merge()

    # Filter through previously merged indexes to create a 5 final indexes 
    # along with 5 corresponding location indexes that hold the information 
    # of the positions of the token data in the final index
    createFinalIndex()

    print(f"{datetime.datetime.now()} Finished Indexing")

