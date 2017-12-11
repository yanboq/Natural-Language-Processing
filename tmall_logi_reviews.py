
# coding: utf-8

# In[23]:

from pandas import Series, DataFrame
from snownlp import SnowNLP
import numpy as np
import jieba
import jieba.analyse
import pprint


# In[24]:

import pandas as pd
df = pd.read_excel('C:/Users/Robin/Google Drive/Code/Data/tmall_logi_reviews.xlsx', encoding='utf-8')
df2 = pd.read_excel('C:/Users/Robin/Google Drive/Code/Data/tmall_logi_products.xlsx')


# In[25]:

df.head(3)


# In[42]:

reviews = df[df['product_id'] == 25423416810]['append_content'].dropna(axis=0)
# reviews = df[df['product_id'] == 38373582263]['content'].dropna(axis=0)
# reviews = df['append_content'].dropna(axis=0)
reviews.shape


# In[43]:

import jieba.analyse
jieba.analyse.set_stop_words('C:/Users/Robin/Google Drive/Code/Project/TextMining/stoplist.txt')
jieba_text = reviews.tolist()
tags = jieba.analyse.extract_tags(str(jieba_text), topK=20, withWeight=True, allowPOS=('a,n'))
word_rank = pd.DataFrame(tags,columns=['word','rank'])
word_rank


# In[44]:

from snownlp import SnowNLP
snow_reviews = ''
for review_temp in reviews:
    snow_reviews = snow_reviews + '。' + str(review_temp)
s = SnowNLP(snow_reviews)
s.summary(15)


# In[7]:

from pandas import Series, DataFrame
review_sentiments = []
for review in reviews:
    try:
        s = SnowNLP(str(review))
        review_sentiment =s.sentiments
        review_sentiments.append(review_sentiment)
    except:
        review_sentiments.append('NaN')
review_obj = DataFrame({'review':reviews,
                        'sentiment':review_sentiments})


# In[10]:

print (review_obj.sort_values('sentiment',axis=0)[0:10])


# In[9]:

rev_len = len(reviews)
n1 = 0
for review_temp in reviews:
    token1  = '麻烦'
    try:
        if token1 in review_temp:
            print (review_temp)
            n1 += 1
    except:
        pass

print (token1,n1,rev_len,n1/rev_len)


# In[20]:

product_id = list(set(df.product_id))

n1_list = []
n1_rate_list = []
len_list = []
pro_name_list = []
for pro_id in product_id:
    pro_reviews = df[df['product_id'] == pro_id]['content'].dropna(axis=0)
    rev_len = len(pro_reviews)
    n1 = 0
    for review_temp in pro_reviews:
        token1  = '男朋友'
        token2 = '老公'
        try:
            if token1 in review_temp:
                n1 += 1
            elif token2 in review_temp:
                n1 += 1
        except:
            pass
    pro_name = df2[df2['product_id'] == pro_id]['name']
    n1_rate = n1/rev_len
    n1_list.append(n1)
    n1_rate_list.append(n1_rate)
    len_list.append(rev_len)
    pro_name_list.append(pro_name)
pro_obj = DataFrame({'pro_id':product_id,
            'token1':n1_rate_list,
            'n1':n1_list,
            'rev_len':len_list,
            'pro_name':pro_name_list})
pprint.pprint (pro_obj.sort_values('token1',axis=0, ascending=False)[0:15])


# In[9]:

all_reviews = df['content'].dropna(axis=0)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



