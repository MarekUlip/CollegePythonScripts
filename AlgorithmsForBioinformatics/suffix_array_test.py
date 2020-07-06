from collections import defaultdict
from itertools import zip_longest, islice
from time import time


def sort_bucket(s, bucket, order):
    d = defaultdict(list)
    for i in bucket:
        key = s[i + order // 2:i + order]
        d[key].append(i)
    result = []
    for k, v in sorted(d.items()):
        if len(v) > 1:
            result += sort_bucket(s, v, 2 * order)
        else:
            result.append(v[0])
    return result


def suffix_array_ManberMyers(s):
    return sort_bucket(s, range(len(s)), 1)


def to_int_keys_best(l):
    """
    l: iterable of keys
    returns: a list with integer keys
    """
    seen = set()
    ls = []
    for e in l:
        if not e in seen:
            ls.append(e)
            seen.add(e)
    ls.sort()
    index = {v: i for i, v in enumerate(ls)}
    return [index[v] for v in l]


def suffix_matrix_best(s):
    """
    suffix matrix of s
    O(n * log(n)^2)
    """
    n = len(s)
    k = 1
    line = to_int_keys_best(s)
    ans = [line]
    while max(line) < n - 1:
        line = to_int_keys_best(
            [a * (n + 1) + b + 1
             for (a, b) in
             zip_longest(line, islice(line, k, None),
                         fillvalue=-1)])
        ans.append(line)
        k <<= 1
    return ans


def suffix_array_best(s):
    """
    suffix array of s
    O(n * log(n)^2)
    """
    n = len(s)
    k = 1
    line = to_int_keys_best(s)
    while max(line) < n - 1:
        line = to_int_keys_best(
            [a * (n + 1) + b + 1
             for (a, b) in
             zip_longest(line, islice(line, k, None),
                         fillvalue=-1)])
        k <<= 1
    return line


text = 'banana'
with open('chr178.txt', 'r') as content_file:
    text = content_file.read()[:10000000]
# print(len(text))
# print(sys.getsizeof(text))
start = time()
suffixt_array = suffix_array_ManberMyers(text)
# for i in range(len(suffixt_array)):
#    suffixt_array[i] +=1
# print(suffixt_array)
"""suffix_array = suffix_array_best(text)
print(suffixt_array)
suffix_array = SuffixArray(text)
print(suffixt_array)"""
print("took {}".format(time() - start))
