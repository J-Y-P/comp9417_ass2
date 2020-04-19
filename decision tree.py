#coding=utf-8
import sys
from copy import deepcopy
import numpy as np
import pandas as pd
import re

# model parameters

sizes = 5000
windows = 7
param_dist = {
         'criterion':['gini','entropy'],
         'max_depth':range(2,8,2),
         'min_samples_leaf':[2,3,5],
         'min_impurity_decrease':[0.1,0.2,0.5]
        }

# read train data
trainData = pd.read_csv(open('training.csv','rb'))
trainData = trainData.dropna()
print(trainData.info())
trainData.head(5)
trainnum = trainData.shape[0]

# distribution
trainData.groupby("topic").count()

# read test data
testData = pd.read_csv(open('test.csv','rb'))
testData = testData.dropna()
#testData["label"] = testData["topic"].apply(lambda x: token_label(x))
print(testData.info())
testData.head(5)

# distribution
testData.groupby("topic").count()

# word counts
rawdata = trainData.append(testData,ignore_index=True)
from copy import deepcopy
def word_freq(df, key):
    wdict={}
    for line in df.loc[:,key]:
        for w in line:
            if(w not in wdict):
                wdict[w] = 1
            else:
                wdict[w] += 1
    return wdict
rawdata["words"] = rawdata["article_words"].apply(lambda x:x.split(","))
rawdata = rawdata[["article_number","words","topic"]]
words_freq_dict = word_freq(rawdata,'words')
list1= sorted(words_freq_dict.items(),key=lambda x:x[1],reverse=True)
print(list1)

# word2vec
def filt_punc(token_list):
    try:
        token_list_ = [word for word in token_list]
        return token_list_
    except Exception as e:
        print(tb.print_exc())
LIST = []
for w in rawdata["words"].values:
    res = filt_punc(w)
    LIST.append(res)

import gensim

# train word2vec
model = gensim.models.Word2Vec(LIST,min_count =1,window =windows,size=sizes)
embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))

print('Found %s word vectors.' % len(embeddings_index))

def doc2vec(arr,model,length = sizes):
    i = 0
    M = []
    
    for w in arr:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:     
        return np.zeros(length+2)
    else:
        return v / np.sqrt((v ** 2).sum())

# label transation
from sklearn.preprocessing import LabelEncoder as le
label_tool = le()
y = label_tool.fit_transform(rawdata.topic.values)

rawdata["y"] = y

neg_label = rawdata[rawdata["topic"]=="IRRELEVANT"]['y'].iloc[0]

def trans_label(x, neg_):
    if x == neg_:
        return 0
    elif x < neg_:
        return x+1
    else:
        return x

rawdata["y"] = rawdata["y"].apply(lambda x:trans_label(x, neg_label))
TrainX = rawdata["words"][:trainnum]
TestX = rawdata["words"][trainnum:]
Trainy = rawdata["y"][:trainnum]
Testy = rawdata["y"][trainnum:]

trainx = np.array([doc2vec(x,model) for x in TrainX])
testx = np.array([doc2vec(x,model) for x in TestX])

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split,GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score,classification_report


kflod = StratifiedKFold(n_splits=10, shuffle = True,random_state=128)
grid = GridSearchCV(DecisionTreeClassifier(),
                        param_dist,
                        cv = kflod,
                        scoring = 'f1_micro',
                        n_jobs = -1)

grid_result = grid.fit(trainx, Trainy)
print("Best: %f using %s" % (grid.best_score_,
        grid.best_params_))

from sklearn.metrics import precision_score, recall_score, f1_score
# trainset
predy1 = grid.predict(trainx)
# testset
predy2 = grid.predict(testx)

print('Train set F1-score: %.4f' %f1_score(Trainy,predy1,average='micro'))
print('Test set F1-score: %.4f' %f1_score(Testy,predy2,average='micro'))

testData["y_pred"] = predy2
testData["y"] = np.array(Testy)

testData = testData[["article_number","topic","y","y_pred"]]

def trans_binary_case(x, pos_):
    if x == pos_:
        return 1
    else:
        return 0
col_name = list(set(list(testData.topic.values)))
col_name.remove('IRRELEVANT')

from copy import deepcopy
for name in col_name:
    pos_label = rawdata[rawdata["topic"]==name]['y'].iloc[0]
    temp = deepcopy(testData)
    temp["y"] = testData["y"].apply(lambda x:trans_binary_case(x, pos_label))
    temp["y_pred"] = testData["y_pred"].apply(lambda x:trans_binary_case(x, pos_label))
    y_temp = temp.y.values
    predy_temp = temp.y_pred.values
    p = precision_score(y_temp,predy_temp)
    r = recall_score(y_temp,predy_temp)
    f1= f1_score(y_temp,predy_temp)
    print(name)
    print('Train set precision: %.4f| recall: %.4f| f1_score: %.4f'%(p,r,f1))

for name in col_name:
    pos_label = testData[testData["topic"]==name]['y'].iloc[0]
    print(pos_label)
    try:
        temp = testData[testData["y_pred"]==pos_label]
        articles = [i for i in temp.article_number.values]
    except:
        articles =[]

    print(name)
    print(articles)
