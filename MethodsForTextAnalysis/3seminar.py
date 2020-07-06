from time import time

"""def create_states(pattern, mistakes):
    n = pattern
    states = [(1,0)]
    for i
    for i in range(1,mistakes+1):
        for j in range(len(pattern)):"""


# states[j*i:[j*i+1,j*i+]]

# [(q1,0)] m -> [(q9,1), (q9,0), (q8,1)]

def infinite_automata(text, pattern, k):
    n = len(pattern)
    states = []
    for i in range(len(text)):
        states.append((1, i, 0))
    success_states = []
    maximum = (n + 1) * (k + 1)
    while len(states) > 0:
        state = states.pop(0)
        if state[1] >= len(text):
            continue
        # print(state)
        if state[0] % (n + 1) == 0:
            success_states.append(state)  # return True, state
            continue
        if state[0] + n + 1 <= maximum:
            states.append((state[0] + n + 1, state[1] + 1, state[2] + 1))  # down
        if state[0] + n + 2 <= maximum:
            states.append((state[0] + n + 2, state[1] + 1, state[2] + 1))  # diagonal_1
            states.append((state[0] + n + 2, state[1], state[2]))  # diagonal_2
        # print("{} {}".format(text[state[1]],pattern[state[0]%n-1]))
        # print((state[0])%(n+1)-1)
        if text[state[1]] == pattern[(state[0]) % (n + 1) - 1]:
            # print((state[0]+1,state[1]+1))
            states.append((state[0] + 1, state[1] + 1, state[2] + 1))
    indexes = []
    for s_t in success_states:
        indexes.append(s_t[1] - s_t[2])
    return set(indexes)


text = "I went to surgrery and had a surgery "
t = ""
for i in range(1):
    t += text
print(len(t))
start = time()
print(infinite_automata(text, "survey", 3))  # "I went to surgrery and had a surgery","survey",5))
print("Took {}".format(time() - start))
print("I went to surgrery"[10:])
