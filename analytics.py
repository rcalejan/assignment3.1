import pickle
from jsonhandling import Posting

f1 = open('masterIndex/index1.pickle', 'rb')
f1p = pickle.load(f1)

f2 = open('masterIndex/index2.pickle', 'rb')
f2p = pickle.load(f2)

f3 = open('masterIndex/index3.pickle', 'rb')
f3p = pickle.load(f3)

f4 = open('masterIndex/index4.pickle', 'rb')
f4p = pickle.load(f4)

f5 = open('masterIndex/index5.pickle', 'rb')
f5p = pickle.load(f5)

print(len(f1p) + len(f2p) + len(f3p) + len(f4p) + len(f5p))
