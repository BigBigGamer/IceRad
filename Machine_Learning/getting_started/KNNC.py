import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


def KNNC(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is not appropriate')
    distances=[]
    for group in data:
        for features in data[group]:
            euclid_dist = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclid_dist,group])
    votes = [ i[1]   for i in sorted(distances)[:k]  ]
    result_votes = Counter(votes).most_common(1)[0][0] 
    confidence = Counter(votes).most_common(1)[0][1] /k
    return result_votes ,confidence


df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999,inplace=True)
#dont need the id column, doesn't give any useful info
df.drop(['id'],1,inplace=True)
full_data = df.astype(float).values.tolist()
#shuffle the data
random.shuffle(full_data)
test_size = 0.2 
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data=full_data[:-int(test_size*len(full_data))]
test_data=full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1]) # add all to dictionary
for i in test_data:
    test_set[i[-1]].append(i[:-1]) # add all to dictionary

correct = 0
total = 0
for group in test_set:
    for data in test_set[group]:
        vote, conf = KNNC(train_set, data, k = 5)
        if group == vote:
            correct+=1
        total+=1
print('Acc: ', correct/total)

