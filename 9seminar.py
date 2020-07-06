import itertools
import random
from copy import deepcopy


def count_overlap_len(str_1: str, str_2: str, min_len=3):
    start = str_1.find(str_2[:min_len], 0)
    while True:
        if start == -1:
            return 0
        if str_2.startswith(str_1[start:]):
            return len(str_1) - start
        start += 1
        start = str_1.find(str_2[:min_len], start)


def find_shortest_common_superstring(sequences, min_len):
    scs = None
    for permutation in itertools.permutations(sequences):
        tmp_scs = permutation[0]
        for i in range(len(permutation) - 1):
            overlap_len = count_overlap_len(permutation[i], permutation[i + 1], min_len)
            tmp_scs += permutation[i + 1][overlap_len:]
        if scs is None or len(scs) > len(tmp_scs):
            scs = tmp_scs
    return scs


def find_shortest_common_superstring_greedy(sequences: list, min_len=1):
    while len(sequences) > 1:
        good_pair = select_good_pair(sequences, min_len=min_len)
        if good_pair[0] is None:
            good_pair = pick_some_pair(sequences)
            sequences.remove(good_pair[0])
            sequences.remove(good_pair[1])
            sequences.append(good_pair[0] + good_pair[1])
            continue
        sequences.remove(good_pair[0])
        sequences.remove(good_pair[1])
        sequences.append(good_pair[0] + good_pair[1][good_pair[2]:])
    return sequences[0]


def pick_some_pair(sequences):
    i = random.randint(0, len(sequences) - 1)
    j = random.randint(0, len(sequences) - 1)
    while i == j:
        j = random.randint(0, len(sequences) - 1)
    return [sequences[i], sequences[j]]


def create_pairs(sequences):
    pairs = []
    for seq_a in sequences:
        for seq_b in sequences:
            if seq_a == seq_b:
                continue
            else:
                pairs.append([seq_a, seq_b])
    return pairs


def select_good_pair(sequences, change_prob=0.5, min_len=1):
    b_seq_a = None
    b_seq_b = None
    b_pair_overlap = 0
    for pair in create_pairs(sequences):
        ovrlap = count_overlap_len(pair[0], pair[1], min_len)
        if ovrlap >= b_pair_overlap:
            if ovrlap == b_pair_overlap:
                if not random.random() <= change_prob:
                    continue
            b_seq_a = pair[0]
            b_seq_b = pair[1]
            b_pair_overlap = ovrlap
    return [b_seq_a, b_seq_b, b_pair_overlap]


# strings = ['BAA', 'AAB', 'BBA', 'ABA', 'ABB', 'BBB', 'AAA', 'BAB']
strings = ['AAA', 'AAB', 'ABB', 'BBB', 'BBA']
# print(find_shortest_common_superstring_greedy(['BAA', 'AAB', 'BBA', 'ABA', 'ABB', 'BBB', 'AAA', 'BAB'],1))
correct_answ = find_shortest_common_superstring(strings[:], 1)
guessed = find_shortest_common_superstring_greedy(deepcopy(strings), 1)
print(correct_answ)
print(len(correct_answ))
print(len(guessed))
while correct_answ != guessed:
    guessed = find_shortest_common_superstring_greedy(deepcopy(strings), 1)
    print(len(guessed))
print('Correct')
# print(count_overlap_len('AAA','AAB',1))
# print(count_overlap_len('AAB','AAA',1))
