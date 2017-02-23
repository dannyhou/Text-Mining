#coding:utf-8

import jieba
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

corpus=list(set([u.split('|')[0] for u in open("xplay_series_link.info",encoding='utf-8').readlines()]))
stop_words=open("stop_words.txt",encoding='utf-8').read()
words_dict={}
words=[]

### avoid stop words
for line in corpus:
    for word in jieba.cut(line):
        if stop_words.count(word)==0:
            words.append(word.strip())

### calc word requency
for word in words:
    words_dict[word]=words_dict.get(word,0)+1

kw_list = sorted(words_dict, key=lambda x: words_dict[x], reverse=True)

###take top N as features
n_features=int(len(kw_list)*0.1)
kw_list=kw_list[:n_features]

### build matrix for features
x_data=[]
for line in corpus:
    features = [0 for _ in range(n_features)]
    for word in jieba.cut(line):
        if stop_words.count(word)==0:
            for i,ind in enumerate(kw_list):
                if word==ind:
                    features[i]=1
    x_data.append(features)

###matrix decomposition
pca=PCA(n_components=100)
X=pca.fit_transform(x_data)

###clustering
kmeans=KMeans(n_clusters=5)
kmeans.fit(X)
X_labels=kmeans.labels_
print(X_labels)
