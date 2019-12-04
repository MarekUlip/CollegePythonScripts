

def join_chars_NKA(chars):
    state_counter = 0
    states = []
    for char in chars:
        states.append([state_counter,char,state_counter+1])
        state_counter+=1
        states.append([state_counter,'',state_counter+1])
        state_counter+=1
    return states[:len(states)-1]

def create_NKA(regexp):
    m = len(regexp)
    ops = []
    edges = []
    for i in range(m):
        lp = i
        if regexp[i] == '(' or regexp[i] == '|' :
            ops.append(i)
        elif regexp[i] == ')':
            ori = ops.pop()

            if regexp[ori] == '|':
                lp = ops.pop()
                edges.append([lp,ori+1])
                edges.append([ori,i])
            elif regexp[ori] == '(':
                lp = ori
            else:
                print("Damn")
                return
        if i<m-1 and regexp[i+1]=='*':
            edges.append([lp,i+1])
            edges.append([i+1,lp])
        if regexp[i] == '(' or regexp[i] == '*' or regexp[i] == ')':
            edges.append([i,i+1])
        return edges
        
def or_NKA(NKA_states):
    state_counter = 0
    states = []
    states.append([0,'',1])
    states.append([0,'',2])
    max_state = 2
    last_state = 0
    for state in NKA_states:
        new_state = state[0]+1
        if new_state<last_state:
            new_state = last_state+1
        states.append([new_state,state[1],new_state+1])
        if last_state < state[2]+1:
            last_state = state[2]+1
        if max_state<state[2]+1:
            max_state = state[2]+1
    states.append([max_state,'',max_state+1])
    states.append([max_state,'',max_state+1])

def parse_regexp(regexp):
    prev_char = ''
    for index, char in enumerate(regexp):
        if char == '|':
            print('Join {} and {}'.format(regexp[index-1],regexp[index+1]))
        if char == '*':
            print('Iterate {}'.format(regexp[index-1]))
    

    
print(parse_regexp('a|bc*'))
#print(join_chars_NKA('ab'))
#print(create_NKA('((a|b)c)*'))