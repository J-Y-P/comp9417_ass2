import pandas as pd 
import numpy
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

#load data
df = pd.read_csv('training.csv')
words_train = df['article_words'].head(9500)
word = []

for i in words_train:
	word.append(i.replace(',',' '))

ff = pd.read_csv('test.csv')
words_test = ff['article_words'].head(500)

for i in words_test:
	word.append(i.replace(',',' '))

#File vectorization
corpus = word
vectorizer=CountVectorizer() 
transformer = TfidfTransformer()  
tf_idf=transformer.fit_transform(vectorizer.fit_transform(corpus))
wordd=vectorizer.get_feature_names()
weight=tf_idf.toarray()


k = 10
t=0

for i in range(9500,10000):
	res = []
	test_x = weight[i]

	for j in range(0,9500):
		train_x = weight[j]
		dist = numpy.sum(numpy.square(train_x - test_x))
		res.append({"topic": df['topic'][j],"distance": dist})

	res= sorted(res,key=lambda item:item['distance'] )
	resk = res[0:k]
	ssum = 0
	result  = {}

	for r in resk:
		result[r['topic']]=0

	for r in resk:
		ssum += r['distance']

	for r in resk:
		result[r['topic']] += 1 - r['distance']/ssum

	topic = max(result, key=lambda x: result[x])

	if (topic == ff['topic'][i-9500]):
		t += 1

print("The test accuracy is: ")
print(t/500)



