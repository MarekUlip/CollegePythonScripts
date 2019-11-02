import sys
from time import time
import collections
import numpy as np

modulator = 10

def process(lines=None):
    ks = ['name', 'sequence', 'optional', 'quality']
    return {k: v for k, v in zip(ks, lines)}

def load_fastq():
    records = []
    n = 4
    with open('ready.fastq', 'r') as fh:
        lines = []
        for line in fh:
            lines.append(line.rstrip())
            if len(lines) == n:
                records.append(process(lines)['sequence'])
                lines = []
    print(len(records))
    return records


def create_suffix_array(pattern):
    whole_pattern = pattern+"0"
    start = time()
    suffix_array = [i for i in range(len(whole_pattern))]#[(whole_pattern[i:],i) for i in range(len(whole_pattern))]
    print("Creating basic suffix array took {}".format(time()-start))
    #print(sys.getsizeof(suffix_array))
    start = time()
    #print(sorted([(whole_pattern[i:],i) for i in range(len(whole_pattern))]))
    #suffix_array = sorted(suffix_array)
    suffix_array = countingSort(suffix_array, pattern,3)
    print(len(suffix_array))
    for index, arr in enumerate(suffix_array):
        #print("{}. {}".format(index,len(arr)))
        quickSort(arr,0,len(arr)-1,whole_pattern,len(whole_pattern))
    new_arr = []
    for arr in suffix_array:
        new_arr.extend(arr)
    suffix_array = new_arr
    #print(new_arr)
    #quickSort(suffix_array,0,len(suffix_array)-1,whole_pattern,len(whole_pattern))
    #print(suffix_array)
    print("Sorting suffix array took {}".format(time()-start))
    return suffix_array



def countingSort(arr, text, prefix=1): 
    n = len(arr) 
 
    count = {}
    
    for i in range(n):
        index = text[i:i+prefix]
        if index == '':
            continue
        count[index] = count.get(index,0) + 1
    indexes = {}
    for index, key in enumerate(sorted(count.keys())):
        indexes[key] = index
    output = [[] for i in range(len(count))]
    i = n-1
    while i>=0: 
        index =  indexes.get(text[i:i+prefix],-1)
        if index == -1:
            i-=1
            continue
        output[index].append(arr[i])
        i -= 1
    return output



def partition(arr,low,high,text,text_len): 
    i =  low-1          # index of smaller element 
    arr_high = arr[high]
    end = text_len-arr_high
    for j in range(low , high):
        switch = True
        index_j = arr[j]
        for k in range(end):
            pivot_char = text[arr_high+k]
            text_char = text[index_j+k]
            if text_char == pivot_char:
                continue
            if text_char > pivot_char:
                switch = False
                break
            break
        
        if switch:
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i]
  
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return  i+1 
  
def quickSort(arr,low,high,text,text_len): 
    if low < high: 
  
        pi = partition(arr,low,high,text,text_len) 
  
        quickSort(arr, low, pi-1,text,text_len) 
        quickSort(arr, pi+1, high,text,text_len) 

def binary_search(arr,l,r, chain,chromosone):
    if r >= l: 
  
        mid = l + (r - l)//2
        text_len = len(chromosone)
        chain_len = len(chain)
        end = text_len - arr[mid]#len(pivot)
        for k in range(end, text_len):
            if k-end >= chain_len:
                break
            chrom_char = chromosone[k]
            text_char = chain[k-end]
            if text_char == chrom_char:
                continue

            if text_char > chrom_char:
                return binary_search(arr, l, mid-1, chain,chromosone)
            else:
                return binary_search(arr, mid + 1, r, chain, chromosone) 
        return mid 
    else: 
        return -1


def create_BWT(suffix_array, text):
    BWT = []
    for i in range(len(suffix_array)):
        if i%1000000 == 0:
            print(i)
        index = suffix_array[i]
        if index == 0:
            BWT.append('0')
        else:
            BWT.append(text[index-1])
    return BWT

def print_suffix_array(suffix_array, text):
    arr = []
    for index in suffix_array:
        arr.append(text[index])
    print(arr)

def count_occurences(suffix_array, text, alphabet_indexes):
    occurences = [0 for i in range (len(alphabet_indexes)+1)]
    for i in suffix_array:
        char = text[i]
        occurences[alphabet_indexes.get(char,-1)+1] +=  1
    return occurences

def count_rank(BWT, alphabet, modulator=modulator):
    rank = []
    counts = {}
    for char in alphabet:
        counts[char] = 0
    #counts = {'A':0,'C':0,'G':0,'T':0}
    #counts = {'a':0,'b':0,'0':0}
    alphabet = alphabet[:len(alphabet)-1]
    for index, char in enumerate(BWT):
        counts[char] +=1
        if index % modulator == 0:
            sub_array = []
            for c in alphabet:
                sub_array.append(counts[c])
            rank.append(sub_array)#([counts[''],counts[char],counts[char],counts[char]])
    return rank

def count_rank_whole(BWT,alphabet):
    rank = []
    counts = {}
    for char in alphabet:
        counts[char] = 0
    #counts = {'A':0,'C':0,'G':0,'T':0}
    #counts = {'a':0,'b':0,'0':0}
    alphabet = alphabet[:len(alphabet)-1]
    for index, char in enumerate(BWT):
        counts[char] +=1
        
        sub_array = []
        for c in alphabet:
            sub_array.append(counts[c])
        rank.append(sub_array)#([counts[''],counts[char],counts[char],counts[char]])
    return rank


def get_rank(char,index, BWT, ranks, alphabet_indexes, modulator=modulator):
    if index%modulator == 0:
        return ranks[index//modulator][alphabet_indexes[char]]
    half = modulator//2
    index_at_rank = index//modulator
    modulo_at_rank = index%modulator
    to_add = 0
    for i in range(modulo_at_rank):
        if BWT[index-i] == char:
            to_add += 1
    return ranks[index_at_rank][alphabet_indexes[char]] + to_add
    """rank = ranks[index_at_rank][alphabet_indexes[char]]
    if modulo_at_rank > half:
        i = (index_at_rank+1)*modulator
        for j in range(index,i):
            if BWT[j] == char:
                rank-=1
    else:
        i = index_at_rank*modulator
        for j in range(i,index+1):
            if BWT[j] == char:
                rank+=1"""
    #return rank

def get_rank_whole(char,index, BWT, ranks, alphabet_indexes, modulator=modulator):
    return ranks[index][alphabet_indexes[char]]
    """half = modulator//2
    index_at_rank = index//modulator
    modulo_at_rank = index%modulator
    rank = ranks[index_at_rank][alphabet_indexes[char]]
    if modulo_at_rank > half:
        i = (index_at_rank+1)*modulator
        for j in range(index,i):
            if BWT[j] == char:
                rank-=1
    else:
        i = index_at_rank*modulator
        for j in range(i,index):
            if BWT[j] == char:
                rank+=1
    return rank"""

def get_range(char, start_index,occurences,alphabet_indexes, end_index=None):
    start_range = 1
    for key in alphabet_indexes.keys():
        if key == char:
            if end_index is None:
                return [start_range+start_index,start_range+occurences[alphabet_indexes[char]+1]]
            else:
                return [start_range+start_index,start_range+end_index]
        start_range += occurences[alphabet_indexes[char]]

def get_index_of(char,index, alphabet, occurences):
    if char == '0':
        return 0
    result_index = 0
    for c in alphabet:
        if c != char:
            result_index += occurences[c]
        else:
            result_index += index
            break
    return result_index

def exists(BWT, occurences, pattern, alphabet, ranks, alphabet_indexes):
    searched_range = get_range(pattern[len(pattern)-1],0,occurences,alphabet_indexes)
    new_indexes = []
    for i in range(len(pattern)-1,0,-1):
        if i == -1:
            break
        
        #print('Searcher range {}'.format(searched_range))
        #print(new_indexes)
        new_indexes.clear()
        #char = pattern[i]
        seeked_char = pattern[i-1]
        for j in range(searched_range[0],searched_range[1]):
            #print(j)
            if BWT[j] == seeked_char:
                #print('Found')
                #print('Getting rank')
                
                new_indexes.append(get_rank(seeked_char,j,BWT,ranks,alphabet_indexes))
        #print(new_indexes)
        if len(new_indexes) == 0:
            return -1
        #print(len(new_indexes))
        #start=time()
        searched_range = get_range(seeked_char,min(new_indexes)-1,occurences,alphabet_indexes,max(new_indexes))
        #print('range search took {}'.format(time()-start))

        #print(searched_range)
    return searched_range





alphabet_indexes = {'a':0,'b':1}
#preprocess_chromosone()
text = 'banana0'
text = 'abaaba0'
alphabet = "ab0"
#with open('chr178.txt', 'r') as content_file:
#    text = content_file.read()[:1000000]
    #alphabe = ACGT0
#print(len(text))
#print(sys.getsizeof(text))
#alphabet = 'ACGT0'
suffix_array = create_suffix_array(text)
#print(suffix_array)
#suffix_array = np.load('suffix_array_real.npy').tolist()
#print_suffix_array(suffix_array,text)
BWT = create_BWT(suffix_array,text)
#np.save('bwt_array',np.array(BWT))
#print(BWT)
occurences = count_occurences(suffix_array,text, alphabet_indexes)
#np.save('occurences',np.array(occurences))
print(occurences)

ranks = count_rank(BWT,alphabet)
#print(ranks)
index_range = exists(BWT,occurences,'aba',alphabet,ranks,alphabet_indexes)
print(index_range)
if index_range != -1:
    for i in range(index_range[0],index_range[1]):
        print(text[suffix_array[i]:])
"""sequences = load_fastq()
found = 0
start = time()
for sequence in sequences:
    
    start = time()
    #print(exists(BWT,occurences,sequence,alphabet,ranks,alphabet_indexes))
    
    if exists(BWT,occurences,sequence,alphabet,ranks,alphabet_indexes) != -1:#binary_search(suffix_array,0,len(suffix_array)-1,sequence+"0",text) > 0:
        found += 1
    print("Took {}".format(time()-start))
    #print(binary_search(suffix_array,0,len(suffix_array)-1,sequence,text))
print("Took {}".format(time()-start))
print(found)"""
#suffix_array = np.array(suffix_array)

#np.save('suffix_array',suffix_array)
alphabet_indexes = {'A':0,'C':1,'G':2,'T':3}
"""suffix_array = np.load('suffix_array.npy').tolist()

text = text[:len(text)]
sequences = load_fastq()
found = 0
for sequence in sequences:
    start = time()
    if binary_search(suffix_array,0,len(suffix_array)-1,sequence+"0",text) > 0:
        found += 1
    #print(binary_search(suffix_array,0,len(suffix_array)-1,sequence,text))
    #print("Took {}".format(time()-start))
print(found)"""
#input()
#print('0'<'AC')
#preprocess_chromosone()

alphabet = "ACGT0"
#with open('chr178.txt', 'r') as content_file:
#    text = content_file.read()+'0'
#    alphabe = ACGT0
#suffix_array = np.load('suffix_array_real.npy').tolist()
#print_suffix_array(suffix_array,text)
BWT = np.load('bwt_array.npy').tolist()
#np.save('bwt_array',np.array(BWT))
occurences = np.load('occurences.npy').tolist()
#np.save('occurences',np.array(occurences))
print(occurences)

#ranks = count_rank(BWT,alphabet)
ranks = np.load('ranks10.npy').tolist()
#np.save('ranks10',np.array(ranks))

sequences = load_fastq()
found = 0
start = time()
for sequence in sequences:
    
    start = time()
    print(exists(BWT,occurences,sequence,alphabet,ranks,alphabet_indexes))
    print("Took {}".format(time()-start))
    if exists(BWT,occurences,sequence,alphabet,ranks,alphabet_indexes) != -1:#binary_search(suffix_array,0,len(suffix_array)-1,sequence+"0",text) > 0:
        found += 1
    #print(binary_search(suffix_array,0,len(suffix_array)-1,sequence,text))
print("Took {}".format(time()-start))
print(found)

#print(ranks[:5])
#np.save('ranks',np.array(ranks))
#print(ranks)
#index_range = exists(BWT,occurences,'aba',alphabet,ranks,alphabet_indexes)"""