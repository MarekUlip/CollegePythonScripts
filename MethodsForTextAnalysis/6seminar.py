class Stemmer:
    def __init__(self):
        self.k = 0
        self.k0 = 0
        self.word = ''

    def is_cons(self, i):
        if self.word[i] == 'a' or self.word[i] == 'e' or self.word[i] == 'i' or self.word[i] == 'o' or self.word[
            i] == 'u':
            return False
        if self.word[i] == 'y':
            if i == self.k0:
                return True
            else:
                return (not self.is_cons(i - 1))
        return True

    def count_m(self, i=0):
        n = 0
        m_string = []
        prev_symbol = ''
        for j in range(i, self.k):
            if prev_symbol != self.word[j]:
                if self.is_cons(j):
                    m_string.append('c')
                else:
                    m_string.append('v')
                prev_symbol = self.word[j]
        if m_string[0] == 'c':
            m_string = m_string[1:]
        if m_string[len(m_string) - 1] == 'v':
            m_string = m_string[:len(m_string) - 1]
        return len(m_string) // 2

    def stem(self, word):
        self.k = len(word)
        self.word = word
