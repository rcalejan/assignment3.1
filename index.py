from bs4 import BeautifulSoup

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


class Posting:
    def __init__(self, docID: int, freq: int, posList: list[int], title: bool, bold: int):
        self.docID = docID      # int: Document Integer ID
        self.freq = freq        # int: frequency of token in Doc
        self.posList = posList  # list[int]: position of the token in the doc 
        self.title = title      # bool: does this token appear in the title in Doc
        self.bold = bold        # int: frequency of token where token was boldened

# need to change tokenizer to preserve HTML IF we want to also include
#  that data in there 

def tokenize(file_content) -> list[str]:
    """Tokenizes content of file and returns a list of the tokens.  Tokens
       are sequencies of numbers, characters, or underscores."""
    file_content = file_content.lower()
    pattern = re.compile(r'\w+')
    tokens = pattern.findall(file_content)
    return tokens

def index(tokens: list[str], startingDocID: int) -> dict[str: Posting]:
    '''
    Returns a dictionary of words in the following form:
    Token : Posting
    '''

# Gonna need to MERGE all the files at 