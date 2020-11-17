''' This will process, train, evaluate, and do all the stuff with the dataset '''
import os
from fakenews import preprocess

fakespath = os.path.abspath('data/Fake.csv')
realspath = os.path.abspath('data/True.csv')
X, Y = preprocess.run(fakespath, realspath)
print(X[:10])
