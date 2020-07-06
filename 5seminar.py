import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

from porterReference import PorterStemmer

stp_wrds = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any', 'are',
            'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 'do', 'does',
            'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
            'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like',
            'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off', 'often',
            'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so',
            'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to',
            'too', 'twas', 'us', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
            'why', 'will', 'with', 'would', 'yet', 'you', 'your']
import numpy as np


def preprocess(text: str, stemmer):
    new_text = []
    text = text.replace('\n', ' ').replace('\r', '')
    for char in text:
        if char.isalnum() or char == ' ':
            new_text.append(char.lower())
    new_text = ''.join(new_text).split()
    new_text = [word for word in new_text if word not in stp_wrds]
    new_text = stem_text(new_text, stemmer)
    return new_text


def stem_text(text, stemmer):
    new_string = []
    if type(text) is str:
        text = text.split()
    for word in text:
        new_string.append(stemmer.stem(word.lower(), 0, len(word) - 1))
    return " ".join(new_string)


# def parse(text):
#    for


# print(preprocess('I went to the zoo'))
text = ''
with open('reut2-000.sgm', mode='r', encoding='utf8') as contet_file:
    text = contet_file.read()
root = BeautifulSoup(text, features='html.parser')
# root = ET.parse('reut2-000.sgm')
# print(root)
title_tag = root.find_all('text')
articles = []
stemmer = PorterStemmer()
for artcl in title_tag:
    body = artcl.find('body')
    if body is None:
        continue
    title = artcl.find('title')
    articles.append([title.text, preprocess(body.text, stemmer)])
articles = np.array(articles)
np.savetxt('articles.csv', articles, delimiter=';', fmt='%s')
"""body_tag = root.find_all('body')
#print("{} {}".format(len(title_tag),len(body_tag)))
articles = []
while title_tag is not None:
    title = title_tag.text
    body = root.find_next('body').text
    articles.append(title,body)
    print(title)
    title_tag = root.find_next('title')"""
