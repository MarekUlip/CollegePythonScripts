states = {'s0': 's', 'l0': 'l', 's1': 'l', 'l1': 's'}
state = 's'
end_state = 'l'


def finite_machine(sequence):
    state = 's'
    for item in sequence:
        state = states[state + item]
    return state == 'l'


def infinite_machine(sequence):
    infinite_states = {"s0": ["s", "l"], "s1": "s", "l1": "e"}
    state = 's'
    queue = [[state + sequence[0], 1]]
    i = 1
    while len(queue) != 0:
        state_info = queue.pop(0)
        new_state = infinite_states.get(state_info[0], -1)
        if new_state != -1 and state_info[1] < len(sequence):
            if type(new_state) is list:
                for s in new_state:
                    queue.append([s + sequence[state_info[1]], state_info[1] + 1])
            else:
                queue.append([new_state + sequence[state_info[1]], state_info[1] + 1])
        if new_state == 'e' and state_info[1] == len(sequence):
            return True
        # print(new_state, state_info)
    return False


# print(finite_machine("01110"))
print(len("0101010101101"))
print(infinite_machine("0101010101101101"))