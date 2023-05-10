from bs4 import BeautifulSoup
from collections import defaultdict
import os
import json
import re
from porter2stemmer import Porter2Stemmer
import pickle
'''
This file is responsible for indexing the documents 
and creating the structure that holds the information
about each term's frequency. 
'''

# GAMEPLAN:
# gonna need to dump to different files at least 3 times
# to ensure we're not storing it all in memory
# Then once all the files are created, 
# merge them into one big file.
current_id = 0
main_index = defaultdict(list)
stemmer = Porter2Stemmer()

class Posting:
    def __init__(self, docID: int, freq: int, posList: list[int], title: bool, bold: bool, header: bool):
        self.docID = docID      # int: Document Integer ID
        self.freq = freq        # int: frequency of token in Doc
        self.posList = posList  # list[int]: position of the token in the doc 
        self.title = title      # bool: does this token appear in the title in Doc
        self.bold = bold        # bool: is this token in a bold tag in the Doc
        self.header = header    # bool: does this token appear in an h1, h2, or h3 tag in the Doc

# need to change tokenizer to preserve HTML IF we want to also include
#  that data in there 
def countFrequencyAndPosition(tokens: list[str]) -> dict[str, int]:
    frequencies = dict()
    for position, word in enumerate(tokens):
        if word not in frequencies:
            frequencies[word] = [1, [position]]
        else:
            frequencies[word][0] += 1
            frequencies[word][1].append(position)
    return frequencies

def tokenize(file_content) -> list[str]:
    """Tokenizes content of file and returns a list of the tokens.  Tokens
       are sequencies of numbers, characters, or underscores."""
    file_content = file_content.lower()
    pattern = re.compile(r'\w+')
    tokens = pattern.findall(file_content)
    porterized = map(stemmer.stem, tokens)
    return porterized

def indexFile(file):
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
            main_index[token].append((Posting(current_id, freq, positions, isTitle, isBold, isHeader)))


# Gonna need to MERGE all the files at
if __name__=='__main__':
    for subdir, dirs, files in os.walk('./developer/DEV'):
        for file in files:
            indexFile(subdir + '/' + file)
    print(main_index)
    f = open('index.json', 'wb')
    f.write(main_index)


