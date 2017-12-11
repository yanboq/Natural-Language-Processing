
# coding: utf-8

# In[2]:

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[3]:

from gensim import corpora


# In[15]:

# import csv
# with open('C:/Users/Yanbo/Documents/Code/Data/shenjianshou/jd_logi_reviews.csv', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile)
#     reviews = []
#     for row in reader:
#         reviews.append(row['评价内容(content)'])


# In[4]:

import pandas as pd
df = pd.read_csv('C:/Users/Yanbo/Documents/Code/Data/shenjianshou/jd_logi_reviews.csv', encoding='utf-8',index_col = '商品ID(product_id)')


# In[9]:

reviews = df.loc[3903182]['评价内容(content)'][0:10]
print (reviews)


# In[18]:

stop_words = []
with open('stoplist.txt','r',encoding='utf-8') as st:
    for i in st.readlines():
        i = i.rstrip()
        stop_words.append(i)


# In[8]:

import jieba.posseg as pseg
attri_reviews = []
for i in reviews:
    words = pseg.cut(i)
    for word, flag in words:
        print('%s %s' % (word, flag))
        attri_reviews.append()


# In[6]:

import jieba
import jieba.analyse


new_reviews = []
for i in reviews:
    data = jieba.cut(i,cut_all=False)
    try:
        texts = [word for word in data if word not in stop_words]
    except AttributeError:
        pass
#     print (texts)
#     temp_text = " ".join(texts)
#     print (temp_text)
    new_reviews.append(texts)

# with open("jd_reviews_cut.txt", "w", encoding='utf-8') as f: 
#     f.write(str(new_reviews))


# In[7]:

# dictionary = corpora.Dictionary(new_reviews)
# dictionary.save('jd_reviews_dict.dict')
dictionary = corpora.Dictionary.load('jd_reviews_dict.dict')


# In[8]:

corpus = [dictionary.doc2bow(text) for text in new_reviews]
# corpora.MmCorpus.serialize('jd_reviews_mm.mm', corpus)
# corpus = corpora.MmCorpus('jd_reviews_mm.mm')


# In[9]:

from gensim.corpora import Dictionary
from gensim.models.lsimodel import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel


# In[55]:

lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=1)
lda.print_topics(5)


# In[56]:

from gensim import corpora, models, similarities
tfidf = models.TfidfModel(corpus)


# In[57]:

corpus_tfidf = tfidf[corpus]


# In[58]:

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5)


# In[70]:

corpus_lsi = lsi[corpus_tfidf]


# In[71]:

lsi.print_topics(5)


# In[10]:

print (new_reviews)


# In[ ]:



