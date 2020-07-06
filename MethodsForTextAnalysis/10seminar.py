from collections import Counter
from copy import deepcopy
from time import time

import numpy as np
from sklearn import preprocessing

from porterReference import PorterStemmer


def dot_product(a, b):
    return np.dot(a, b)


def dot_self_made(a, b):
    res = 0
    for i in len(a):
        res += a[i] + b[i]
    return res


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)  # preprocessing.normalize(vector)


def normalize_matrix(matrix):
    return preprocessing.normalize(matrix, axis=0)


def euclidean_lenght(vector):
    return np.sqrt(np.dot(vector, vector))


def cosine_similarity(a, b):
    return dot_product(a, b) / euclidean_lenght(a)


def document_similiarity_finder(vector, matrix, num_of_docs):
    scores = []
    scores_cos = []
    for i in range(num_of_docs):
        for j in range(num_of_docs):
            if i == j:
                continue
            scores.append([j, dot_product(matrix[:, i], matrix[:, j])])
            scores_cos.append([j, cosine_similarity(matrix[:, i], matrix[:, j])])
        print(sorted(scores, key=lambda tmp: tmp[1], reverse=True)[0])
        print(sorted(scores_cos, key=lambda tmp: tmp[1], reverse=True)[0])
        scores.clear()
        scores_cos.clear()


def get_most_similiar_document(vector, matrix, num_of_docs):
    scores = []
    scores_cos = []
    vector.append(0)
    for i in range(num_of_docs):
        scores.append([i, dot_product(matrix[:, i], vector)])
        scores_cos.append([i, cosine_similarity(matrix[:, i], vector)])

    print(sorted(scores, key=lambda tmp: tmp[1], reverse=True)[0])
    print(sorted(scores_cos, key=lambda tmp: tmp[1], reverse=True)[0])


def get_query_vector(query, term_map: dict, num_of_words, stemmer):
    query = stem_words(query, stemmer)
    query_vector = [0 for i in range(num_of_words)]
    word_counts = dict(Counter(query))
    print(num_of_words)
    print(word_counts)
    for word in query:
        index = term_map.get(word, -1)
        if index >= 0:
            print(index)
            query_vector[index] = word_counts.get(word, 0)
    print(Counter(query_vector))
    return query_vector


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

# print(normalize_vector([0,1,1]))
word_count = len(term_map.keys())
print(euclidean_lenght([0, 1, 1, 1]))
vector = get_query_vector('measure state art', term_map, word_count, stemmer)
# tf = np.transpose(normalize_matrix(np.transpose(tf_matrix)))
tf = normalize_matrix(tf_matrix)
# document_similiarity_finder([],tf,doc_count)
get_most_similiar_document(vector, tf, doc_count)
tf_idf_matrix = normalize_matrix(tf_idf_matrix)
vector = get_query_vector('measure state art', term_map, word_count, stemmer)
get_most_similiar_document(vector, tf_idf_matrix, doc_count)
