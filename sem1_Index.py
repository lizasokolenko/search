
# coding: utf-8

# ## Семинар 1 Индекс
# 
# ## Intro

# ### работа с файлами и папками

# In[1]:


import os

curr_dir = os.getcwd()
filepath = os.path.join(curr_dir, 'test.txt')


# ### os.path  
# путь до файла

# In[2]:


# возвращает полный путь до папки/файла по имени файла / папки
print(os.path.abspath(filepath))


# возвращает имя файла / папки по полному ти до него
print(os.path.basename(filepath))


# проверить существование директории - True / False
print(os.path.exists(curr_dir))


# ### os.listdir  
# возвращает список файлов в данной директории

# In[2]:


os.listdir(curr_dir)


# При обходе файлов не забывайте исключать системные директории, такие как .DS_Store

# ### os.walk
# root - начальная директория  
# dirs - список поддиректорий (папок)   
# files - список файлов в этих поддиректориях  

# In[4]:


for root, dirs, files in os.walk('Users\lizasokolenko\Downloads\friends\\'):
    for name in files:
        print(os.path.join(root, name))


# > __os.walk__ возвращает генератор, это значит, что получить его элементы можно только проитерировавшись по нему  
# но его легко можно превратить в list и увидеть все его значения

# In[2]:


list(os.walk('C:\\Users\\lizasokolenko\\Downloads\\friends\\'))


# ### чтение файла 

#   

# Напоминание про enumerate:    
# > При итерации по списку вы можете помимо самого элемента получить его порядковый номер    
# ``` for i, element in enumerate(your_list): ...  ```    
# Иногда для получения элемента делают так -  ``` your_list[i] ```, не надо так

# ##  Индекс 
# 
# Сам по себе индекс - это просто формат хранения данных, он не может осуществлять поиск. Для этого необходимо добавить к нему определенную метрику. Это может быть что-то простое типа булева поиска, а может быть что-то более специфическое или кастомное под задачу.
# 
# Давайте посмотрим, что полезного можно вытащить из самого индекса.    
# По сути, индекс - это информация о частоте встречаемости слова в каждом документе.   
# Из этого можно понять, например:
# 1. какое слово является самым часто употребимым / редким
# 2. какие слова встречаются всегда вместе - так можно парсить твиттер, fb, форумы и отлавливать новые устойчивые выражения в речи
# 3. как эти документы кластеризуются по N тематикам согласно словам, которые в них упоминаются 

# ## __Задача__: 
# 
# **Data:** Коллекция субтитров сезонов Друзьей. Одна серия - один документ.
# 
# **To do:** Постройте небольшой модуль поискового движка, который сможет осуществлять поиск по коллекции документов.
# На входе запрос и проиндексированная коллекция (в том виде, как посчитаете нужным), на выходе отсортированный по релевантности с запросом список документов коллекции. 
# 
# Релизуйте:
#     - функцию препроцессинга данных
#     - функцию индексирования данных
#     - функцию метрики релевантности 
#     - собственно, функцию поиска
# 
# [download_friends_corpus](https://yadi.sk/d/yVO1QV98CDibpw)

# Напоминание про defaultdict: 
# > В качестве multiple values словаря рекомендую использовать ``` collections.defaultdict ```                          
# > Так можно избежать конструкции ``` dict.setdefault(key, default=None) ```

# In[3]:


### _check : в коллекции должно быть около 165 файлов
files = list(os.listdir('C:\\Users\\lizasokolenko\\Downloads\\friends\\'))
files


# С помощью обратного индекса посчитайте:  
# 
# 
# a) какое слово является самым частотным
# 
# b) какое самым редким
# 
# c) какой набор слов есть во всех документах коллекции
# 
# d) какой сезон был самым популярным у Чендлера? у Моники?
# 
# e) кто из главных героев статистически самый популярный? 
# 

# In[4]:


from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()


# In[5]:


from nltk.tokenize import WordPunctTokenizer
import re


# In[6]:


def preproc(file):
    with open(file, 'r', encoding='utf-8') as f:
        t = f.read()
    t = re.sub(r'[\.!\(\)?,;:\-\"\ufeff]', r'', t)
    text = WordPunctTokenizer().tokenize(t)
    preproc_text = ''
    for w in text:
        new_w = morph.parse(w)[0].normal_form + ' '
        preproc_text += new_w
        #preproc_text.append(morph.parse(w)[0].normal_form)
    return preproc_text


# In[7]:


for f in files:
    path = 'C:\\Users\\lizasokolenko\\Downloads\\friends\\{}'.format(f)
    preproc(path)


# In[8]:


import time


# In[9]:


texts_words = []
for file in files:
    t = file
    fpath = 'C:\\Users\\lizasokolenko\\Downloads\\friends\\{}'.format(file)
    ws = preproc(fpath)
    texts_words.append(ws)
    if len(texts_words) % 10 == 0:
        print(f'{len(texts_words)} done')
        time.sleep(2)
#texts_words


# In[11]:


texts_words


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer

#docs = ['why hello there', 'omg hello pony', 'she went there? omg']
vec = CountVectorizer()
X = vec.fit_transform(texts_words)


# In[11]:


import pandas as pd

df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=files)


# In[12]:


df.head()


# In[13]:


df_transpose = df.transpose()
df_transpose.head()


# ### Обратный индекс
# С помощью обратного индекса посчитайте:  
# 
# 
# a) какое слово является самым частотным
# 
# b) какое самым редким
# 
# c) какой набор слов есть во всех документах коллекции
# 
# d) какой сезон был самым популярным у Чендлера? у Моники?
# 
# e) кто из главных героев статистически самый популярный? 
# 

# In[14]:


#df_transpose = df_transpose.replace([i for i in (1, 1000)], 1)
df_transpose = df_transpose.applymap(lambda x: 1 if x > 0 else 0)


# In[15]:


df_transpose['sum'] = df_transpose.sum(axis=1)


# In[16]:


df_transpose.head()


# самые частотные слова и самые редкие (внизу)

# In[114]:


df_transpose.sort_values(by=['sum'], ascending=False)


# In[17]:


df_transpose.loc[['фиби','моника','чендлера','рэйчел','джой','росс'], :]

