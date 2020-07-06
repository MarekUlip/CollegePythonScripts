import numpy as np

seq_a = 'TATGTCATGC'
seq_b = 'TACGTCAGC'
table_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}


# MRIP
# opakujici sekvence jsou malymi pismeny

def global_alignment(seq_a, seq_b):
    p_func = np.array([[0, 4, 2, 4, 8], [4, 0, 4, 2, 8], [2, 4, 0, 4, 8], [4, 2, 4, 0, 8], [8, 8, 8, 8, 10000]])
    len_a = len(seq_a) + 1
    len_b = len(seq_b) + 1
    matrix = np.zeros((len_a, len_b), dtype=int)
    for i in range(1, len_a):
        matrix[i, 0] = matrix[i - 1, 0] + p_func[table_map.get('-'), table_map.get(seq_a[i - 1])]
    for i in range(1, len_b):
        matrix[0, i] = matrix[0, i - 1] + p_func[table_map.get(seq_b[i - 1]), table_map.get('-')]

    for i in range(1, len_a):
        for j in range(1, len_b):
            matrix[i, j] = min([matrix[i - 1, j] + p_func[table_map.get(seq_a[i - 1]), table_map.get('-')],
                                matrix[i, j - 1] + p_func[table_map.get('-'), table_map.get(seq_b[j - 1])],
                                matrix[i - 1, j - 1] + p_func[
                                    table_map.get(seq_a[i - 1]), table_map.get(seq_b[j - 1])]])
    # print(matrix)
    return matrix[len_a - 1, len_b - 1], matrix


def local_alignment(seq_a, seq_b):
    p_func = np.array(
        [[2, -4, -4, -4, -6], [-4, 2, -4, -4, -6], [-4, -4, 2, -4, -6], [-4, -4, -4, 2, -6], [-6, -6, -6, -6, 100000]])
    len_a = len(seq_a) + 1
    len_b = len(seq_b) + 1
    matrix = np.zeros((len_a, len_b), dtype=int)
    for i in range(1, len_a):
        matrix[i, 0] = 0  # matrix[i-1,0]+p_func[table_map.get('-'),table_map.get(seq_a[i-1])]
    for i in range(1, len_b):
        matrix[0, i] = 0  # matrix[0,i-1]+p_func[table_map.get(seq_b[i-1]),table_map.get('-')]

    for i in range(1, len_a):
        for j in range(1, len_b):
            matrix[i, j] = max([matrix[i - 1, j] + p_func[table_map.get(seq_a[i - 1]), table_map.get('-')],
                                matrix[i, j - 1] + p_func[table_map.get('-'), table_map.get(seq_b[j - 1])],
                                matrix[i - 1, j - 1] + p_func[table_map.get(seq_a[i - 1]), table_map.get(seq_b[j - 1])],
                                0])
    # print(matrix)
    return max([max(row) for row in matrix]), matrix


def resolve_diff_global(diff):
    if diff == 0:
        return 'M'
    elif diff == 2:
        return 'R'
    elif diff == 4:
        return 'R'
    elif diff == 8:
        return 'I'


def resolve_diff_local(diff):
    if diff == 2:
        return 'M'
    elif diff == -4:
        return 'I'
    elif diff == -6:
        return 'R'


def resolve_index(index, i, j):
    if index == 0:
        return i - 1, j
    elif index == 1:
        return i, j - 1
    if index == 2:
        return i - 1, j - 1


def trace_back_global(matrix, seq_a, seq_b):
    p_func = np.array([[0, 4, 2, 4, 8], [4, 0, 4, 2, 8], [2, 4, 0, 4, 8], [4, 2, 4, 0, 8], [8, 8, 8, 8, 10000]])
    i = len(seq_a)
    j = len(seq_b)
    traceback = ""
    while i != 0 and j != 0:
        # print("{} {}".format(i,j))
        surrounding = [matrix[i - 1, j] + p_func[table_map.get(seq_a[i - 1]), table_map.get('-')],
                       matrix[i, j - 1] + p_func[table_map.get('-'), table_map.get(seq_b[j - 1])],
                       matrix[i - 1, j - 1] + p_func[table_map.get(seq_a[i - 1]), table_map.get(seq_b[j - 1])]]
        edit_vals = [p_func[table_map.get(seq_a[i - 1]), table_map.get('-')],
                     p_func[table_map.get('-'), table_map.get(seq_b[j - 1])],
                     p_func[table_map.get(seq_a[i - 1]), table_map.get(seq_b[j - 1])]]
        # print(surrounding)
        best_index = surrounding.index(min(surrounding))
        i, j = resolve_index(best_index, i, j)
        traceback += resolve_diff_global(edit_vals[best_index])
    return traceback[::-1]


def get_highest_num_index(matrix):
    highest_in_row = [max(row) for row in matrix]
    max_val = max(highest_in_row)
    i = np.where(highest_in_row == max_val)[0][0]
    j = np.where(matrix[i] == max_val)[0][0]
    return [i, j]


def check_surrounding(matrix, i, j):
    if i - 1 < 0 or j - 1 < 0:
        return False
    if sum([matrix[i - 1, j], matrix[i, j - 1], matrix[i - 1, j - 1]]) == 0:
        return False
    return True


def trace_back_local(matrix, seq_a, seq_b):
    p_func = np.array(
        [[2, -4, -4, -4, -6], [-4, 2, -4, -4, -6], [-4, -4, 2, -4, -6], [-4, -4, -4, 2, -6], [-6, -6, -6, -6, 100000]])
    i, j = get_highest_num_index(matrix)
    traceback = ""
    can_trace = check_surrounding(matrix, i, j)
    safety = True
    while can_trace:
        # print("{} {}".format(i,j))
        surrounding = [matrix[i - 1, j] + p_func[table_map.get(seq_a[i - 1]), table_map.get('-')],
                       matrix[i, j - 1] + p_func[table_map.get('-'), table_map.get(seq_b[j - 1])],
                       matrix[i - 1, j - 1] + p_func[table_map.get(seq_a[i - 1]), table_map.get(seq_b[j - 1])]]
        edit_vals = [p_func[table_map.get(seq_a[i - 1]), table_map.get('-')],
                     p_func[table_map.get('-'), table_map.get(seq_b[j - 1])],
                     p_func[table_map.get(seq_a[i - 1]), table_map.get(seq_b[j - 1])]]
        # print(surrounding)
        best_index = surrounding.index(max(surrounding))
        i, j = resolve_index(best_index, i, j)
        traceback += resolve_diff_local(edit_vals[best_index])
        can_trace = check_surrounding(matrix, i, j)
        if not can_trace and safety:
            can_trace = True
            safety = False
    return traceback[::-1]


# print(global_alignment(seq_b,seq_a))
print("Global alignment result {}".format(global_alignment(seq_b, seq_a)[0]))
print("Traceback global {}".format(trace_back_global(global_alignment(seq_b, seq_a)[1], seq_b, seq_a)))
seq_a = 'TATATGCGGCGTTT'
seq_b = 'GGTATGCTGGCGCTA'
print("Local alignment result {}".format(local_alignment(seq_b, seq_a)[0]))
print("Traceback local {}".format(trace_back_local(local_alignment(seq_b, seq_a)[1], seq_b, seq_a)))
