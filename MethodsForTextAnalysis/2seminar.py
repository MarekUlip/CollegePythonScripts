from time import time


def brute_force(text_to_search, pattern):
    i = 0
    counter = 0
    indexes = []
    while i < len(text_to_search):
        if text_to_search[i] == pattern[0]:
            matcher = True
            for index, item in enumerate(pattern):
                if index + i >= len(text_to_search):
                    matcher = False
                    break
                if text_to_search[index + i] != item:
                    matcher = False
                    break
            if matcher:
                indexes.append(i)
                counter += 1
        i += 1
    return indexes


def DKA_init(pattern):
    prefix = ""
    alphabet = ["A", "C", "G", "T"]  # ,"N","W", "S","R", 'K','Y','V','D','M','H','B']
    Q = [prefix]
    E = {prefix: prefix}
    for char in pattern:
        prefix += char
        Q.append(prefix)
    for q in Q:
        for a in alphabet:
            if q + a in Q:
                E[q + a] = q + a  # .append((q,a,q+a))
            else:
                qa = q + a
                for i in range(len(qa) + 1):
                    p = qa[len(qa) - i:]  # qa[:len(qa)-i]
                    for sub_rule in Q:
                        if sub_rule == p:
                            E[q + a] = p
                            break
                    """if p in str(Q):
                        E[q+a] = p#.append(q,a,p)
                        break"""
    # print(Q)
    # print(E)
    return E


def DKA_search(text_to_search, rules: dict, pattern):
    state = ""
    indexes = []
    for index, char in enumerate(text_to_search):
        state = rules.get(state + char, '')
        if state == pattern:
            indexes.append(index - len(pattern) + 1)
    return indexes


def BMH_preprocess(alphabet, pattern):
    T = {}
    pattern_len = len(pattern)
    for i in alphabet:
        T[i] = pattern_len
    for i in range(pattern_len - 1):
        T[pattern[i]] = pattern_len - 1 - i
    return T


def BMH(pattern, text_to_search):
    indexes = []
    T = BMH_preprocess("ACGT", pattern)  # NWSRKYVDMHB",pattern)
    skip = 0
    text_len = len(text_to_search)
    pattern_len = len(pattern)
    while text_len - skip >= pattern_len:
        # print(skip)
        i = pattern_len - 1
        while text_to_search[skip + i] == pattern[i]:
            if i == 0:
                indexes.append(skip)
                break
            i -= 1
        skip += T.get(text_to_search[skip + pattern_len - 1], pattern_len)
        # if T[text_to_search[skip+pattern_len-1]] == 0:
        #    skip += 1
    return indexes


pattern = "GCAGAGAG"
text = "GCATCGCAGAGAGTATACAGTACG"
with open('dna - Copy.txt', 'r') as content_file:
    text = content_file.read().replace('\n', '')

start = time()
print(len(DKA_search(text, DKA_init(pattern), pattern)))
print("DKA took {}".format(time() - start))
start = time()
print(len(BMH(pattern, text)))
print("Boyer-Moore-Horspool took {}".format(time() - start))
start = time()
print(len(brute_force(text, pattern)))
print("Brute force took {}".format(time() - start))

pattern = "GCAGAGAGGACGAAGCCCAACGAGGGGCAAAGGGCGCAGAGAGGACGAAGCCCAACGAGGGGCAAAGGGCGCAGAGAGGACGAAGCCCAACGAGGGGCAAAGGGCGCAGAGAGGACGAAGCCCAACGAGGGGCAAAGGGCGCAGAGAGGACGAAGCCCAACGAGGGGCAAAGGGC"
start = time()
print(len(DKA_search(text, DKA_init(pattern), pattern)))
print("DKA took {}".format(time() - start))
start = time()
print(len(BMH(pattern, text)))
print("BMH took {}".format(time() - start))
start = time()
print(len(brute_force(text, pattern)))
print("Brute force took {}".format(time() - start))
