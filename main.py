''' This will process, train, evaluate, and do all the stuff with the dataset '''
from fakenews import preprocess

X, Y = preprocess.run()
print(X[:10])
