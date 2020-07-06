def is_cons(i, word):
    if word[i] == 'a' or word[i] == 'e' or word[i] == 'i' or word[i] == 'o' or word[i] == 'u':
        return False
    if word[i] == 'y':
        if i == k0:
            return True
        else:
            return (not is_cons(i - 1, word))
    return True


def count_m(word, i=0):
    n = 0
    m_string = []
    prev_symbol = ''
    for j in range(i, k):
        if prev_symbol != word[j]:
            if is_cons(j):
                m_string.append('c')
            else:
                m_string.append('v')
            prev_symbol = word[j]
    if m_string[0] == 'c':
        m_string = m_string[1:]
    if m_string[len(m_string) - 1] == 'v':
        m_string = m_string[:len(m_string) - 1]
    return len(m_string) // 2


def m(word):
    """m() measures the number of consonant sequences between k0 and j.
    if c is a consonant sequence and v a vowel sequence, and <..>
    indicates arbitrary presence,

       <c><v>       gives 0
       <c>vc<v>     gives 1
       <c>vcvc<v>   gives 2
       <c>vcvcvc<v> gives 3
       ....
    """
    n = 0
    i = 0
    while 1:
        if i > j:
            return n
        if not is_cons(i, word):
            break
        i = i + 1
    i = i + 1
    while 1:
        while 1:
            if i > j:
                return n
            if is_cons(i, word):
                break
            i = i + 1
        i = i + 1
        n = n + 1
        while 1:
            if i > j:
                return n
            if not is_cons(i, word):
                break
            i = i + 1
        i = i + 1
