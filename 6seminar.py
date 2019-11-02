import numpy as np
import copy
from time import time

def create_list_of_substrings(text, w=3):
    substrings = []
    for i in range(0,len(text)-w+1):
        substrings.append(text[i:i+w])
    return substrings

def generate_substrings_rec(alphabet,strings,string, leng, max_len):
    if leng==max_len:
        return string
    for char in alphabet:
        stri = generate_substrings_rec(alphabet,strings,string+char,leng+1,max_len)
        if stri is not None:
            strings.append(stri)

def generate_substrings(alphabet):
    substrings = []
    for char in alphabet:
        for char2 in alphabet:
            for char3 in alphabet:
                substrings.append(char+char2+char3)
    return substrings

def score_substrings(string,substrings, scoring_matrix, mappings, w=3, threshold=11):
    score_map = {}
    for i in range(len(substrings)):
        s1 = substrings[i]
        score = 0
        for k in range(w):
            score += scoring_matrix[mappings[string[k]]][mappings[s1[k]]]
        if score >= threshold:
            score_map[s1] = score
    return score_map
    #print(sorted(score_map.values(),reverse=True)[:10])

def get_sequence_start_index(sub_sequence, sequence, w=3,skip=0):
    skipped = 0
    print(sub_sequence)
    print(sequence)
    for i in range(len(sequence)-w-1):
        if sequence[i:i+w] == sub_sequence and skip == skipped:
            return i
        else:
            skipped+=1
    return -1

def extend_alignment(index,database_index, sequence, database_sequence,scoring_matrix, mappings, direction=1,w=3):
    drop_off = 0
    if direction == 1:
        index += w
        database_index += w
    else:
        index -= 1
        database_index -= 1
    seq_len = len(sequence)
    dat_seq_len = len(sequence)
    extended_alignment = []
    score = 0
    while drop_off >=-2:
        #print(drop_off)
        if index >= seq_len or index < 0 or database_index >= dat_seq_len or database_index < 0:
            return ["".join(extended_alignment),score]
        drop_off = scoring_matrix[mappings[sequence[index]]][mappings[database_sequence[database_index]]]
        if drop_off >= -2:
            score+=drop_off
            extended_alignment.append(sequence[index])
            index+=direction
            database_index+=direction
        else:
            break
    return ["".join(extended_alignment),score]

def get_base_subsequence_score(sub_sequence, scoring_matrix, mappings):
    score = 0
    for char in sub_sequence:
        score += scoring_matrix[mappings[char]][mappings[char]]
    return score




def find_hsp(sub_sequence, sequence, database, scoring_matrix, mappings, base_score, sub_sequence_index, w=3):
    index = get_sequence_start_index(sub_sequence,sequence,w,0)
    print(index)
    skip = 0
    hsp = ['',0]
    #while index >= 0:
    for database_sequence in database:
        database_index = get_sequence_start_index(sub_sequence,database_sequence,w,0)
        #for i in range(len(database_sequence)-w-1):
        if database_index >=0:
            right_alignment = extend_alignment(index,i,sequence,database_sequence,scoring_matrix,mappings,1,w)
            left_alignment = extend_alignment(index,i,sequence,database_sequence,scoring_matrix,mappings,-1,w)
            score = base_score + right_alignment[1]+left_alignment[1]
            print("{} {}".format(score, hsp))
            if score > hsp[1]:
                hsp = [left_alignment[0]+sub_sequence+right_alignment[0], score]
                print(hsp)
            skip+=1
        #index = get_sequence_start_index(sub_sequence,sequence,w,skip)
    return hsp

def get_msp(sequence, database, scoring_matrix, mappings, substrings, w=3):
    sub_sequences = create_list_of_substrings(sequence,w)
    msp = ['',0]
    for index, sub_sequence in enumerate(sub_sequences):
        scores = score_substrings(sub_sequence,substrings,scoring_matrix,mappings,w)
        #print(scores)
        #print(len(scores))
        for key, score in scores.items():
            hsp = find_hsp(key,sequence,database,scoring_matrix,mappings,score,index,w)
            if hsp[1] >= msp[1]:
                #print(hsp)
                msp = hsp
            #print('HSP processed {} {} {}'.format(key,hsp, score))
        #print('sub sequence processed.')
    return msp

        


def create_map(string):
    string_map = {}
    for index, char in enumerate(string):
        string_map[char] = index
    return string_map

mappings = create_map('A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V'.split())
text = []
with open('blosum62.txt', 'r') as content_file:
    text = content_file.read().splitlines()
texts = [line.split() for line in text]
texts[0].append('')
#[print(line) for line in texts]
texts = np.array(texts)
#print(texts)
blosum = texts[1:21,1:21].astype('int')
database = ['DAPCQEHKRGWPNDC']#'SGQYTKQSPVSSS','ADEGILEHKMWP','VLNDCQEGHILRS','DAPCQEHKRGWPNDC']
#print(texts)
strings = []
generate_substrings_rec('A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V '.split(),strings,'',0,3)
#print(len(strings))
#print(strings[-25:])
substrings = generate_substrings('A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V '.split())
#msp = get_msp('SGQYTKQSPVSSS',database, blosum,mappings,substrings,3)
#test_seq = 'NDCLEHKMWWAWDWNDCLEHKMWWAWDWNDCLEHKMWWAWDWNDCLEHKMWWAWDWNDCLEHKMWWAWDWNDCLEHKMWWAWDWNDCLEHKMWWAWDWLEHFPTWY'
test_seq = 'YANCLEHKMGS'
#test_seq = 'NDCLEHKMWWAWDWND'
start = time()
msp = get_msp(test_seq,database, blosum,mappings,substrings,3)
print('len {} took {}'.format(len(test_seq),time()-start))
print(msp)
#print("len {}, expected {}".format(len(generate_substrings('A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V '.split())),20**3))
#score_substrings('LEH',generate_substrings('A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V '.split()),blosum,mappings)
#substrings = create_list_of_substrings('alfara')
#print(substrings)
#print('A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *'.split())