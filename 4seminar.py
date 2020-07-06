import copy
import sys
from time import time

import numpy as np


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
    whole_pattern = pattern + "0"
    start = time()
    suffix_array = [i for i in range(len(whole_pattern))]  # [(whole_pattern[i:],i) for i in range(len(whole_pattern))]
    print("Creating basic suffix array took {}".format(time() - start))
    # print(sys.getsizeof(suffix_array))
    start = time()
    # print(sorted([(whole_pattern[i:],i) for i in range(len(whole_pattern))]))
    # suffix_array = sorted(suffix_array)
    suffix_array = countingSort(suffix_array, whole_pattern, 3)
    print(len(suffix_array))
    for index, arr in enumerate(suffix_array):
        print("{}. {}".format(index, len(arr)))
        quickSort(arr, 0, len(arr) - 1, whole_pattern, len(whole_pattern))
    new_arr = []
    for arr in suffix_array:
        new_arr.extend(arr)
    suffix_array = new_arr
    # print(new_arr)
    # quickSort(suffix_array,0,len(suffix_array)-1,whole_pattern,len(whole_pattern))
    # print(suffix_array)
    print("Sorting suffix array took {}".format(time() - start))
    return suffix_array


def compare_quicksorts(pattern):
    whole_pattern = pattern + "0"
    start = time()
    suffix_array = [i for i in range(len(whole_pattern))]  # [(whole_pattern[i:],i) for i in range(len(whole_pattern))]
    print("Creating basic suffix array took {}".format(time() - start))
    # print(sys.getsizeof(suffix_array))
    tst_arr = copy.deepcopy(suffix_array)
    start = time()
    quickSort(tst_arr, 0, len(tst_arr) - 1, whole_pattern, len(whole_pattern))
    print("old quicksort took {}".format(time() - start))
    # print(tst_arr)
    # tst_arr = copy.deepcopy(suffix_array)
    # print(tst_arr)
    # start = time()
    # quickSortIterative(tst_arr,0,len(tst_arr)-1,whole_pattern,len(whole_pattern))
    # print('iterative quicksort took {}'.format(time()-start))
    # print(tst_arr)
    # quickSort(suffix_array,0,len(suffix_array)-1,whole_pattern,len(whole_pattern))
    # print(suffix_array)
    # print("Sorting suffix array took {}".format(time()-start))


def countingSort(arr, text, prefix=1):
    n = len(arr)
    # output = [[]] * (n)

    count = {}

    for i in range(n):
        index = text[i:i + prefix]
        if index == '':
            continue
        # print(index)
        count[index] = count.get(index, 0) + 1

    # sorted_x = sorted(count.items(), key=lambda kv: kv[0])
    indexes = {}
    for index, key in enumerate(sorted(count.keys())):
        indexes[key] = index
    # count = collections.OrderedDict(sorted_x)
    output = [[] for i in range(len(count))]  # [[]] * (len(count))
    # prev_val = 0
    # for key, value in count.items():
    #    count[key] = value + prev_val
    #    prev_val = count[key]
    # print(count)
    # Change count[i] so that count[i] now contains actual
    # position of this digit in output array
    # for i in range(1,10):
    #	count[i] += count[i-1]

    # Build the output array
    i = n - 1
    # print(output)
    while i >= 0:
        index = indexes.get(text[i:i + prefix], -1)
        if index == -1:
            i -= 1
            continue
        output[index].append(arr[i])
        # print(output)
        # output[ count[index] - 1] = arr[i]
        # count[index] -= 1
        i -= 1

    # Copying the output array to arr[], 
    # so that arr now contains sorted numbers 
    # i = 0
    # for i in range(0,len(arr)):
    #    arr[i] = output[i]
    # print(count)
    # print(arr)
    # print(output)
    return output

    # Method to do Radix Sort 


def radix_sort(arr):
    # Find the maximum number to know number of digits 
    max1 = max(arr)

    # Do counting sort for every digit. Note that instead 
    # of passing digit number, exp is passed. exp is 10^i 
    # where i is current digit number 
    exp = 1
    while max1 / exp > 0:
        countingSort(arr, exp)
        exp *= 10


def partition(arr, low, high, text, text_len):
    i = low - 1  # index of smaller element
    arr_high = arr[high]
    end = text_len - arr_high  # len(pivot)
    # switch = True
    for j in range(low, high):
        switch = True
        index_j = arr[j]
        for k in range(end):
            pivot_char = text[arr_high + k]
            text_char = text[index_j + k]
            # if index >= text_len:
            #    break
            if text_char == pivot_char:
                continue
            if text_char > pivot_char:  # pivot[k]:
                switch = False
                break
            break

        if switch:
            # print("switching")
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]

        """pivot = text[arr[high]:]
        if text[arr[j]:] < pivot: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i]"""

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quickSort(arr, low, high, text, text_len):
    if low < high:
        pi = partition(arr, low, high, text, text_len)

        quickSort(arr, low, pi - 1, text, text_len)
        quickSort(arr, pi + 1, high, text, text_len)


def quickSortIterative(arr, l, h, text, text_len):
    stack = []

    stack.append(l)
    stack.append(h)

    while len(stack) > 0:

        h = stack.pop()
        l = stack.pop()

        if h - l + 1 < 10:
            insert_sort(arr, l, h, text, text_len)
        else:
            p = partition(arr, l, h, text, text_len)

            if p - 1 > l:
                stack.append(l)
                stack.append(p - 1)

            if p + 1 < h:
                stack.append(p + 1)
                stack.append(h)


def insert_sort(arr, l, h, text, text_len):
    for x in range(l, h + 1):
        key = arr[x]
        end = text_len - key
        y = x - 1
        while y > -1:  # and arr[y]> key:
            switch = True
            arr_y = arr[y]
            for k in range(end):
                pivot_char = text[key + k]
                text_char = text[arr_y + k]
                if text_char == pivot_char:
                    continue
                if text_char > pivot_char:  # pivot[k]:
                    switch = False
                    break
                break
            if switch:
                break
            arr[y + 1] = arr[y]
            y = y - 1
        arr[y + 1] = key


def binary_search(arr, l, r, chain, chromosone):
    if r >= l:

        mid = l + (r - l) // 2

        # arr_high = arr[high]
        text_len = len(chromosone)
        chain_len = len(chain)
        arr_mid = arr[mid]
        # end = text_len - arr[mid]-1#len(pivot)
        for k in range(arr_mid, text_len):
            if k - arr_mid >= chain_len:
                # print('breaking')
                break
            chrom_char = chromosone[k]
            text_char = chain[k - arr_mid]
            # print(chromosone[k:])
            # print(mid)
            # print("{} {}".format(chrom_char,text_char))
            if text_char == chrom_char:
                # print('con')
                continue

            if text_char < chrom_char:
                return binary_search(arr, l, mid - 1, chain, chromosone)
            else:
                return binary_search(arr, mid + 1, r, chain, chromosone)
                # print("{} {}".format(chromosone[arr[mid]:], chain))
        """ if chromosone[arr[mid]:] == chain: 
            return mid 
          
        elif chromosone[arr[mid]:] > chain: 
            return binary_search(arr, l, mid-1, chain,chromosone) 
        else: 
            return binary_search(arr, mid + 1, r, chain, chromosone) """
        # print(text[arr[mid]:])
        # print(text[end:])
        return mid



    else:
        return -1


def preprocess_chromosone():
    with open('chr17.fa', 'r') as content_file:
        text = content_file.read().replace('\n', '')
    new_text = []
    text = text.upper()
    print(len(text))
    print('upper text created')
    print(text[:100])
    allowed = ['A', 'C', 'G', 'T']
    for index, char in enumerate(text):
        if index % 1000000 == 0:
            print(index)
        if char in allowed:
            new_text.append(char)
    # print(new_text)
    with open('chr178.txt', 'w', encoding='utf8', newline='') as content_file:
        content_file.write(''.join(new_text))


# preprocess_chromosone()
# text = 'banana'
text = 'abaaba'
# with open('chr178.txt', 'r') as content_file:
#   text = content_file.read()
print(len(text))
print(sys.getsizeof(text))
suffix_array = create_suffix_array(text)
# compare_quicksorts(text[:1000000])
# print(suffix_array)

# suffix_array = np.array(suffix_array)

# np.save('suffix_array_real',suffix_array)

#

# text = text[:len(text)]
text += '0'
sequence = 'ba0'
print(suffix_array)
print(binary_search(suffix_array, 0, len(suffix_array) - 1, sequence, text))

with open('chr178.txt', 'r') as content_file:
    text = content_file.read()
suffix_array = np.load('suffix_array_real.npy').tolist()
input()
text += '0'
sequences = load_fastq()
found = 0
for sequence in sequences:
    start = time()
    if binary_search(suffix_array, 0, len(suffix_array) - 1, sequence + "0", text) > 0:
        found += 1
    # print(binary_search(suffix_array,0,len(suffix_array)-1,sequence,text))
    # print("Took {}".format(time()-start))
print(found)
# input()
# print('0'<'AC')
# preprocess_chromosone()
