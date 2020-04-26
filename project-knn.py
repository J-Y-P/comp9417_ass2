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

#File vectorization,use TF-IDF for feature extraction
corpus = word
vectorizer=CountVectorizer() 
transformer = TfidfTransformer()  
tf_idf=transformer.fit_transform(vectorizer.fit_transform(corpus))
wordd=vectorizer.get_feature_names()
weight=tf_idf.toarray()

#Parameter k can be modified
k = 6
t=0
topic_predict = {'ARTS CULTURE ENTERTAINMENT':0,'BIOGRAPHIES PERSONALITIES PEOPLE':0,'DEFENCE':0,'DOMESTIC MARKETS':0,'FOREX MARKETS':0,'HEALTH':0,'MONEY MARKETS':0,'SCIENCE AND TECHNOLOGY':0,'SHARE LISTINGS':0,'SPORTS':0,'IRRELEVANT':0}
predict_true = {'ARTS CULTURE ENTERTAINMENT':0,'BIOGRAPHIES PERSONALITIES PEOPLE':0,'DEFENCE':0,'DOMESTIC MARKETS':0,'FOREX MARKETS':0,'HEALTH':0,'MONEY MARKETS':0,'SCIENCE AND TECHNOLOGY':0,'SHARE LISTINGS':0,'SPORTS':0,'IRRELEVANT':0}

for i in range(9500,10000):
	res = []
	test_x = weight[i]

	for j in range(0,9500):
		train_x = weight[j]
		#get the distance array
		dist = numpy.sum(numpy.square(train_x - test_x))
		res.append({"topic": df['topic'][j],"distance": dist})

	#sort and get the former k distance
	res= sorted(res,key=lambda item:item['distance'] )
	resk = res[0:k]
	if (i == 9500):
		print(resk)
	ssum = 0
	result  = {}

	for r in resk:
		result[r['topic']]=0

	for r in resk:
		ssum += r['distance']

	#weighted sum 
	for r in resk:
		result[r['topic']] += 1 - r['distance']/ssum

	topic = max(result, key=lambda x: result[x])
	topic_predict[topic] += 1

	if (topic == ff['topic'][i-9500]):
		predict_true[topic] += 1
		t += 1

print(topic_predict)
print(predict_true)
print("The test accuracy is: ")
print(t/500)



