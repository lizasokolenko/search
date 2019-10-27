import nltk
from math import log
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
import re
import pickle
import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import operator
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
import logging
logging.basicConfig(filename='preprocessing.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',level=logging.INFO)

russian_stopwords = stopwords.words("russian")

def preprocessing(text):
    text = str(text)
    text = re.sub(r'[0-9\ufeff]', r'', text)
    normal_text = ''
    for t in tokenizer.tokenize(text):
                t = morph.parse(t)[0]
                if t.normal_form not in russian_stopwords and t.normal_form != 'это' and t.normal_form != 'весь' and not re.match(r'[0-9A-Za-z]', t.normal_form):
                    normal_form = t.normal_form + ' '
                    normal_text += normal_form
    return normal_text

all_text = []
numb = 0
with open('quora_question_pairs_rus.csv', encoding='utf-8') as qd:
    qd = csv.reader(qd, delimiter=',')
    for raw in qd:
        numb += 1
        if numb < 502 and numb >= 2:
            all_text.append(raw)

with open("TF-IDF_Texts", 'wb') as f:
    pickle.dump((all_text), f)

with open("TF-IDF_Texts", 'rb') as f:
    all_text = pickle.load(f)

#tf-idf
doc = []
for el in all_text:
    doc.append(preprocessing(el[2]))

len(doc)
corpus = doc
vect = TfidfVectorizer()
X = vect.fit_transform(corpus)
X.shape
logging.info('Made tf-idf dataframe')

def searchTFIDF(qu):
    qu = vect.transform([preprocessing(qu)]).transpose()
    e = X.dot(qu)
    df_tfIdf = pd.DataFrame(e.toarray(), index=doc)
    dict_tfIdf = {}
    for i in range(len(doc)):
        dict_tfIdf[all_text[i][2]] = df_tfIdf[0][i]
        sorted_tfidf = sorted(dict_tfIdf.items(), key=operator.itemgetter(1))
    for elem in sorted_tfidf[-10:]:
        print(elem[0], elem[1])

searchTFIDF('красивая жизнь')

#BM25

numb = 0
docs = []
for i in all_text:
    if preprocessing(i[2]) != '':
        docs.append(preprocessing(i[2]))
    else:
        docs.append('')

k = 2.0
b = 0.75
N = len(docs)
l = 0
for i in all_text:
    l += len(i[2].split())
avgdl = l / N

def bm25(D, Q):
    D = D.split()
    score = 0
    n = 0
    for doc in docs:
        if Q in doc:
            n += 1
    IDF = log((N - n + 0.5) / (n + 0.5))
    TF = 0
    for word in D:
        if word == Q:
            TF += 1
    TF = TF / len(D)
    score += IDF * ((TF * (k + 1)) / (TF + (k * (1 - b + (b * (len(D) / avgdl))))))
    return score

unique_words = {}
uniq = []
for el in docs:
    for word in el.split(' '):
        if word not in unique_words:
            unique_words[word] = 0
for el in unique_words:
    uniq.append(el)
print(uniq)

df = pd.DataFrame(columns=uniq, index=docs)
print(len(uniq))
print(len(docs))

bm25_matrix = np.zeros((500, 1465))  # create zero matrix

for doc_id, text in enumerate(docs):
    for word_id, word in enumerate(df.columns):
        try:
            score = bm25(text, word)
            bm25_matrix[doc_id, word_id] = float(score)

        except:
            pass

logging.info('Made bm25 dataframe')

with open("Indexed_bm.pickle", 'wb') as f:
    pickle.dump((unique_words, docs, bm25_matrix, all_text), f)

with open("Indexed_bm.pickle", 'rb') as f:
    unique_words, docs, bm25_matrix, all_text = pickle.load(f)

def hotenc(query):
    query = preprocessing(query)
    query = query.split()
    vec = []
    for i in unique_words:
        if i in query:
            vec.append(1)
        else:
            vec.append(0)
    df_query = pd.DataFrame(vec, index=unique_words)
    return df_query

def searchBM25(qu):
    qu = hotenc(preprocessing(qu))
    e = bm25_matrix.dot(qu)
    df_bm = pd.DataFrame(e, index=docs)
    dict_bm = {}
    for i in range(len(docs)):
        dict_bm[all_text[i][2]] = df_bm[0][i]
    sorted_bm = sorted(dict_bm.items(), key=operator.itemgetter(1))
    return sorted_bm[-10:]

searchBM25("красивая жизнь")

#fasttext
fast_model = 'model.model'
model = KeyedVectors.load(fast_model)
wv = model.wv
logging.info('Made fasttext dataframe')

def lookup(doc, wv):
    d = preprocessing(doc)
    checked = []

    for word in d:
        try:
            word in wv
        except AttributeError:
            continue
        checked.append(wv[word])

    vec = np.mean(checked, axis=0)
    return vec

def cos_sim(v1, v2):
    return np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_dist(text1, text2, wv):
    t1 = lookup(text1, wv)
    t2 = lookup(text2, wv)
    dist = cos_sim(t1, t2)
    return dist

corpus = 'quora_question_pairs_rus.csv'

with open(corpus, 'r', encoding='utf-8') as f:
    f.read()

def get_data(corpus, wv, stop=5000):
    indexed = []
    id_to_text = {}
    query_to_dupl_id = {}
    counter = 0

    with open(corpus, 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        for line in r:

            if line[0] == '':
                continue

            _id, text, query, isduplicate = line
            id_to_text[_id] = text

            if isduplicate == '1':
                query_to_dupl_id[query] = _id

            indexed.append(lookup(text, wv))

            counter += 1
            if counter >= stop:
                break
    return indexed, id_to_text, query_to_dupl_id

indexed, id_to_text, query_to_dupl_id = get_data(corpus, wv, stop=50000)

with open("Indexed_FT.pickle", 'wb') as f:
    pickle.dump((indexed, id_to_text, query_to_dupl_id), f)

with open("Indexed_FT.pickle", 'rb') as f:
    indexed, id_to_text, query_to_dupl_id = pickle.load(f)

def search(query, wv, indexed):
    query_v = lookup(query, wv)
    result = {}
    for i, doc_vector in enumerate(indexed):
        score = cos_sim(query_v, doc_vector)
        if type(score) is np.float32:
            result[i] = score

    return sorted(result.items(), key=lambda x: x[1], reverse=True)

def word(text):
    ind = search(text, wv, indexed)[:10]
    for el in ind:
        print(id_to_text['{}'.format(el[0])], el[1])

word("красивая жизнь")