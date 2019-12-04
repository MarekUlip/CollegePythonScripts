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

def stem_word(word, stemmer:PorterStemmer):
    return stemmer.stem(word.lower(),0,len(word)-1)

def list_intersection(lists):
    lists = sorted(lists, key=len)
    final_list = lists[0]
    for i in range(1,len(lists)):
        final_list = [item for item in final_list if item in lists[i]]
    return final_list

def list_conjuction(lists):
    final_list = []
    for lst in lists:
        final_list.extend(lst)
    return set(final_list)

def list_exclude(lst, not_list):
    return [item for item in lst if item not in not_list]

def enhanced_query(e_query, term_dict, stemmer):
    q = e_query.lower().split()
    segments = []
    not_segments = []
    segment = []
    not_segment = []
    if stem_word(q[0],stemmer) in term_dict:
        segment.append(term_dict[stem_word(q[0],stemmer)])
    w_c = len(q)
    i=1
    while i < w_c:
        if q[i] == 'and' and q[i+1] != 'not':
            word = stem_word(q[i+1],stemmer)
            if word in term_dict:
                segment.append(term_dict[word])
            i+=2
        elif q[i] == 'and' and q[i+1] == 'not':
            word = stem_word(q[i+2],stemmer)#skips not
            if word in term_dict:
                not_segment.append(term_dict[word])
            i+=3
        elif q[i] == 'or':
            segments.append(segment)
            not_segments.append(not_segment)
            segment = []
            not_segment = []
            word = stem_word(q[i+1],stemmer)
            if word in term_dict:
                segment.append(term_dict[word])
            i+=2
    if len(segment) >= 1:
        segments.append(segment)
        not_segments.append(not_segment)
    final_and_lists = []
    for i in range(len(segments)):
        sgmnt_lst = list_intersection(segments[i])
        nt_segment = not_segments[i]
        if len(nt_segment) > 0:
            nt_segment = [item for sublist in nt_segment for item in sublist]
            sgmnt_lst = list_exclude(sgmnt_lst,nt_segment)
        final_and_lists.append(sgmnt_lst)
        #sgmnt_lst = list_exclude(sgmnt_lst,not_segments)
    return list_conjuction(final_and_lists)
    
        

stemmer = PorterStemmer()

articles = np.loadtxt('articles.csv',dtype=str,delimiter=';')
term_dict = create_term_dict(create_basic_term_array(articles))
query_s = 'measure'
e_query = 'file and state or measure'#'measure or state and file'

print(query(query_s, term_dict,stemmer))
print(sorted(enhanced_query(e_query,term_dict,stemmer)))
print(len(term_dict))

#tst = [[5,4,8,2],[5,9,8,6]]
#print(list_conjuction(tst))
