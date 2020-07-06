class City:
    def __init__(self, x, y, letter, num):
        self.x = x
        self.y = y
        self.letter = letter
        self.num = num


A = City(60, 200, 'A', 0)
B = City(80, 200, 'B', 1)
C = City(80, 180, 'C', 2)
D = City(140, 180, 'D', 3)
E = City(20, 160, 'E', 4)
F = City(100, 160, 'F', 5)
G = City(200, 160, 'G', 6)
H = City(140, 140, 'H', 7)
I = City(40, 120, 'I', 8)
J = City(100, 120, 'J', 9)
K = City(180, 100, 'K', 10)
L = City(60, 80, 'L', 11)
M = City(120, 80, 'M', 12)
N = City(180, 60, 'N', 13)
O = City(20, 40, 'O', 14)
P = City(100, 40, 'P', 15)
Q = City(200, 40, 'Q', 16)
R = City(20, 20, 'R', 17)
S = City(60, 20, 'S', 18)
T = City(160, 20, 'T', 19)
cities = [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T]
cities_point_list = []
# cities_dict = {0:[60, 200], 1: [80, 200], 2: [80, 180], 3: [140, 180], 4: [20, 160], 5: [100, 160], 6: [200, 160], 7: [140, 140], 8: [40, 120], 9: [100, 120], 10: [180, 100], 11: [60, 80], 12: [120, 80], 13: [180, 60], 14: [20, 40], 15: [100, 40], 16: [200, 40], 17: [20, 20], 18: [60, 20], 19: [160, 20]}
cities_letter_to_num = {}
for item in cities:
    cities_letter_to_num[item.letter] = item.num
    cities_point_list.append([item.x, item.y])

print(cities_point_list)
"""string = 'cities = ['
for i in range(65,85):
    string+= "'{}', ".format(chr(i))#+chr(i)+', '

print(string)"""
