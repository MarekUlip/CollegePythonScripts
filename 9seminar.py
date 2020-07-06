from copy import deepcopy
from time import time

import numpy as np

from porterReference import PorterStemmer


def create_tf_matrix(documents):
    term_index_map = {}
    last_index = 0
    matrix = [[0 for i in range(len(documents))]]
    for index, document in enumerate(documents):
        for word in document[1].split():
            if word not in term_index_map:
                term_index_map[word] = last_index
                last_index += 1
                matrix.append([0 for i in range(len(documents))])
            matrix[term_index_map[word]][index] += 1
    print(len(matrix))
    return matrix, term_index_map


def count_idf(tf_matrix, doc_count):
    idf = [0 for i in range(len(tf_matrix))]
    for index, row in enumerate(tf_matrix):
        count = 0
        for col in row:
            if col > 0:
                count += 1
        idf[index] = np.log10(doc_count / (count + 1))
    return idf


def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = np.array(tf_matrix).astype('float64')
    for index, idf in enumerate(idf_matrix):
        tf_idf_matrix[index] *= idf
    return tf_idf_matrix


def stem_words(words, stemmer: PorterStemmer):
    if type(words) is str:
        words = words.split()
    return [stemmer.stem(word.lower(), 0, len(word) - 1) for word in words]


def find_relative_documents(tf_idf_matrix, query, term_map: dict, stemmer: PorterStemmer, doc_count):
    words = stem_words(query, stemmer)
    words = [term_map.get(word, None) for word in words]
    words = [word for word in words if word is not None]
    scores = []
    for i in range(doc_count):
        score = 0
        for w_id in words:
            score += tf_idf_matrix[w_id, i]
        scores.append([i, score])
    scores = sorted(scores, key=lambda tup: tup[1], reverse=True)
    return scores[:10]


articles = np.loadtxt('articles.csv', dtype=str, delimiter=';')
doc_count = len(articles)
start = time()
tf_matrix, term_map = create_tf_matrix(articles)
print("creating tf took: {}".format(time() - start))
start = time()
idf = count_idf(tf_matrix, doc_count)
print("creating idf took: {}".format(time() - start))
start = time()
tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf)
print("creating tf-idf took: {}".format(time() - start))
start = time()
stemmer = PorterStemmer()
print(find_relative_documents(tf_idf_matrix, "measure state art", term_map, stemmer, doc_count))
print("searching took: {}".format(time() - start))
