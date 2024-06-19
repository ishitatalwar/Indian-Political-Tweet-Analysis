#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd 
from textblob import TextBlob 
import plotly.graph_objects as go # Fixed the typo here
import plotly.express as xp


# In[99]:


modi = pd.read_csv("modi_reviews.csv")
rahul = pd.read_csv("rahul_reviews.csv")


# In[100]:


modi.head(3)


# In[101]:


modi.shape


# In[102]:


rahul.head(3)


# In[103]:


rahul.shape


# In[104]:


modi['Tweet'][0]


# In[105]:


TextBlob(modi['Tweet'][0]).sentiment


# In[106]:


rahul['Tweet'][10]


# In[107]:


TextBlob(rahul['Tweet'][10]).sentiment


# In[109]:


modi.info()


# In[110]:


modi['Tweet'] = modi['Tweet'].astype(str)
rahul['Tweet'] = rahul['Tweet'].astype(str)
def find_polarity(review):
    return TextBlob(review).sentiment.polarity


# In[111]:


find_polarity('I hate living without coding')


# In[112]:


find_polarity('I like living without coding')


# In[113]:


find_polarity('I belive in rahul he will win')


# In[114]:


modi['Polarity'] = modi['Tweet'].apply(find_polarity)
rahul['Polarity'] = rahul['Tweet'].apply(find_polarity)


# In[115]:


modi


# In[118]:


rahul.head()


# In[119]:


import numpy as np


# In[120]:


modi['Label'] = np.where(modi['Polarity']>0, 'positive', 'negative')
modi['Label'][modi['Polarity']==0]='Neutral'
rahul['Label'] = np.where(rahul['Polarity']>0, 'positive', 'negative')
rahul['Label'][rahul['Polarity']==0]='Neutral'


# In[121]:


modi.head(20)


# In[122]:


rahul.head(20)


# In[124]:


neutral_modi = modi[modi['Polarity']== 0.0000]

remove_neutral_modi = modi['Polarity'].isin(neutral_modi['Polarity'])
modi.drop(modi[remove_neutral_modi].index, inplace=True)


# In[125]:


print(neutral_modi.shape)
print(modi.shape) 


# In[126]:


neutral_rahul = rahul[rahul['Polarity']== 0.0000]

remove_neutral_rahul = rahul['Polarity'].isin(neutral_rahul['Polarity'])
rahul.drop(rahul[remove_neutral_rahul].index,inplace=True)
print(neutral_rahul.shape)
print(rahul.shape)


# In[127]:


print(modi.shape) 
print(rahul.shape)


# In[128]:


# modi 
np.random.seed(10)
remove_n = 8481
drop_indices = np.random.choice(modi.index, remove_n, replace=False)
df_modi = modi.drop(drop_indices)


# In[131]:


# rahul
np.random.seed(10)
remove_n = 367
drop_indices = np.random.choice(rahul.index, remove_n , replace=False)
df_rahul = rahul.drop(drop_indices)


# In[132]:


print(df_modi.shape)
print(df_rahul.shape)


# In[151]:



modi_count


# In[152]:


rahul_count

modi_count = df_modi.groupby('Label').count()
neg_modi = (modi_count['Polarity'][0]/1000) * 100
pos_modi = (modi_count['Polarity'][1]/1000) * 100

# In[153]:


rahul_count = df_rahul.groupby('Label').count()
neg_rahul = (rahul_count['Polarity'][0]/1000) * 100
pos_rahul = (rahul_count['Polarity'][1]/1000) * 100


# In[154]:


politicians = ['Modi', 'Rahul']

neg_list = [neg_modi, neg_rahul]
pos_list = [pos_modi, pos_rahul]
fig = go.Figure(
data = [
    go.Bar(name = 'Negative', x = politicians, y=neg_list),
    go.Bar(name = 'Positive', x = politicians, y=pos_list)
    
]
)
fig.update_layout(barmode='group')
fig.show()


# In[ ]:




