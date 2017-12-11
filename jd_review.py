
# coding: utf-8

# ##### encoding=utf-8
# import jieba
# import jieba.analyse
# import pandas as pd
# import numpy as np
# import gensim
# import pickle
# from gensim import models
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import CountVectorizer
# from gensim.corpora import Dictionary
# from gensim.models.lsimodel import LsiModel
# from gensim.models.ldamodel import LdaModel
# from gensim.models.tfidfmodel import TfidfModel

# In[4]:

content = open('jd_reviews.txt','rb').read()


# In[5]:

tags = jieba.analyse.extract_tags(content, topK=20, withWeight=True)
word_rank = pd.DataFrame(tags,columns=['word','rank'])
word_rank


# In[6]:

sentences = []
with open('jd_reviews.txt', 'rb') as f:
    sentences += [list(jieba.cut(line.strip())) for line in f]


# In[7]:

print (sentences [20:100])


# In[8]:

model = gensim.models.Word2Vec(sentences, 
                               size=100, 
                               window=5, 
                               min_count=5, 
                               workers=4)


# In[9]:

for k, s in model.most_similar(positive=["鼠标"]):
    print (k, s)


# In[10]:

def find_relationship(a, b, c):
    d, _ = model.most_similar(positive=[c, b], negative=[a])[0]
    str = ("给定“{}”与“{}”，“{}”和“{}”有类似的关系").format(a, b, c, d)
    print (str)

find_relationship("女生", "鼠标", "男生")


# In[11]:

def save(data, file):
    
    fo = open(file, 'wb')
    pickle.dump(data, fo, protocol=2)
    fo.close()


# In[12]:

dataset_name = 'jd_logi'
dictionary = Dictionary(sentences)
save(dictionary, '%s_dict.pkl' % dataset_name)
corpus = [dictionary.doc2bow(text) for text in sentences]


# In[13]:

# id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')


# In[14]:

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=1)
lda.print_topics(5)

