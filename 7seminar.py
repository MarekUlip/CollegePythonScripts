import numpy as np
from porterReference import PorterStemmer
import time

def create_basic_term_array(articles):
    term_array = []
    for index, article in enumerate(articles):
        tmp_array = []
        for word in article[1].split():
            term_doc = [word,index]
            if term_doc not in tmp_array:
                tmp_array.append(term_doc)
        term_array.extend(tmp_array)
    print(len(term_array))
    """tmp_array = []
    print('now sort')
    print(len(term_array))
    for term_doc in term_array:
        if term_doc not in tmp_array:
            tmp_array.append(term_doc)
    print('donio')"""
    return term_array

def create_term_dict(term_array):
    term_dict = {}
    for term in term_array:
        arr = term_dict.get(term[0],[])
        arr.append(term[1])
        term_dict[term[0]] = arr
    return term_dict


def query(words,term_dict, stemmer):
    lists = []
    words = stem_words(words,stemmer)
    for word in words:
        if word in term_dict:
            lists.append(term_dict[word])
    return list_intersection(lists)

def stem_words(words, stemmer:PorterStemmer):
    if type(words) is str:
        words = words.split()
    return [stemmer.stem(word.lower(),0,len(word)-1) for word in words]

def list_intersection(lists):
    lists = sorted(lists, key=len)
    final_list = lists[0]
    for i in range(1,len(lists)):
        final_list = [item for item in final_list if item in lists[i]]
    return final_list

def list_conjuction(lists):
    final_list = []
    for lst in lists:
        final_list = set(final_list.extend(lst))
    return final_list

stemmer = PorterStemmer()

articles = np.loadtxt('articles.csv',dtype=str,delimiter=';')
term_dict = create_term_dict(create_basic_term_array(articles))
query_s = 'measure state art'

start = time.time()
print(query(query_s, term_dict,stemmer))
print(time.time() - start)
print(len(term_dict))
