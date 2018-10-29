import time


def create_towns(town_count):
    a = []
    for i in range(town_count):
        a.append(str(i))
    return a


def show_permutations(prefix, field):
    if len(field) == len(prefix):
        print(prefix)
        return
    for index, value in enumerate(field):
        if value == "":
            continue
        field[index] = ""
        show_permutations(prefix + value, field)
        field[index] = value


def show_permutations_for(field, item):
    for index, value in enumerate(field):
        if value == item:
            field[index] = ""
            show_permutations(value, field)
            break

start = time.time()
#show_permutations("", create_towns(3))
show_permutations_for(create_towns(11), "0")
print(time.time()-start)












def factorial(n):
    if n == 0:
        return 1
    return n*factorial(n-1)


def factorial_field(work_field):
    print(work_field)

    w_field = list(work_field)
    if len(work_field) == 0:
        return work_field
    for item in work_field:
        w_field.append(item)
        # results.append(factorial_field(field[1:], w_field))


def create_sub_array(array):
    res = []
    while len(array) > 0:

        for item in enumerate(array):
            for i in array[1:]:
                res.append()

def test(towns,depth = 0):
    result = []
    for index, value in enumerate(towns[depth:]):
        result.append([])

def tst(towns, field, depth = 1):
    print(towns[2:])
    #print(depth)
    if len(towns) == 1:
        field.append(towns[0])
    else:
        field.append(tst(towns[depth:], field, depth+1))

def aatst(towns):
    size = len(towns)-1
    first = towns[0]
    res = []
    for i in range(1, size):
        tmp = [first]
        for j in range(i, size):
            tmp.append(towns[j])
        res.append(tmp)
        print("tst {}".format(tmp))
    print(res)
# aatst(create_towns(3))



def atst(prev, towns):
    if len(towns) == 0:
        return [].append(prev)
    else:
        print(type(atst(towns[0], towns[1:])))
        # atst(towns[0], towns[1:]).insert(0, prev)

towns = create_towns(3)
#print(atst(towns[0], towns[1:]))

"""for i in a:
    for j in a[1:]:
        for k in a[2:]:"""



result = []
#print(factorial_field(create_towns(3)))
#print(result)

def create_permutations(towns):
    permutations = []
    temp = []
    beggining = 0
    work_towns = list(towns)
    for item in towns:
        beggining = item

        temp.append(item)
        temp.append(item)
    temp.append(0)
    print(temp)



def add_item(field,towns):
    if len(towns) == 0:
        return
    add_item()

#print(create_permutations(create_towns(5)))





