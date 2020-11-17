''' This will process, train, evaluate, and do all the stuff with the dataset '''
from fakenews import preprocess
from fakenews import extractor
import gensim


news, labels = preprocess.run('data/Fake.csv', 'data/True.csv')
preprocess.truncate_news(news)
model = extractor.extract(news, 5, 1)
print(news[0][:10])
