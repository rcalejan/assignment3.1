from index import tokenize
from collections import defaultdict

'''
This is where the ranking algorithm goes.
Look into tf-idf ranking
'''

# for each word in the query
# get list of Postings
# Put each posting object into a dictionary with key as document_id and value as list containing Posting object
#

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

# stem the query 
def askUser():
    print("==========  Welcome to GHoogle!  ==========\n")
    query = input("What do you want to know?: ")


def tokenizeQuery(content):
    fileDict = defaultdict(list)
    tokens = tokenize(content)
    for token in tokens:
        files = []
        if token not in stop_words:
            pass
        #     TODO - files = getFiles(token)
        # for file in files:    
        #     fileDict[file.docID].append(file)

def rankFiles(fileDict):
    def sumFrequencies(docID):
        sum = 0
        for posting in fileDict[docID]:
            sum += posting.freq
        return sum
    ranked = sorted(fileDict.keys(), lambda x: -sumFrequencies(x))
    return ranked

if __name__ == '__main__':
    query = askUser()
    tokens = tokenizeQuery(query)
    print(tokens)