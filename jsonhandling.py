'''
This holds the file where we're going to dump the dictionary 
holding all the delicious inverted index information from a directory
'''

import json
from json import JSONEncoder
from index import Posting


def merge_JsonFiles(filename):
    result = list()
    for f1 in filename:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open('counseling3.json', 'w') as output_file:
        json.dump(result, output_file)


example = {
    "apple": [Posting(1, 20, [1, 5, 9], True, 3), Posting(2, 15, [1, 5, 9], True, 3)],
    "orange": [Posting(1, 13, [6, 2, 3], False, 3)],
    "banana": [Posting(1, 28, [4, 7, 8], False, 2), Posting(2, 25, [1, 5, 9], True, 21)]
}

# print(PostingEncoder().encode(example))

# postingJSONData = json.dumps(, indent=4, cls=PostingEncoder)

apple = Posting(1, 20, [1, 5, 9], True, 3)
print(apple.toJSON())

# s = json.dumps(foo) # raises TypeError with "is not JSON serializable"

s = json.dumps(apple.__dict__) # s set to: {"x":1, "y":2}

# with open('result.json', 'w') as fp:
#     json.dump(apple, fp)

# print("Decode JSON formatted Data")
# postingJSON = json.loads(postingJSONData)
# print(postingJSON)
