import csv
dataset = []

with open('testData.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        dataset.append(row)

#print(dataset)
outlook = ["","Rainy","Overcast","Sunny"]
temperature = ["","Hot", "Mild", "Cool"]
humidity = ["","High","Normal"]
windy = ["","TRUE", "FALSE"]
names = ["outlook","temperature","humidity","windy"]

"""outlook = ["","?","Rainy","Overcast","Sunny"]
temperature = ["","?","Hot", "Mild", "Cool"]
humidity = ["","?","High","Normal"]
windy = ["","?","TRUE", "FALSE"]"""

all_vars = [outlook, temperature, humidity, windy]
possibilities = []

def create_variations(depth,field):
    if depth==len(all_vars):
        possibilities.append(field)
        return
    
    for item in all_vars[depth]:
        f = [a for a in field]
        f.append(item)
        create_variations(depth+1,f)
    


def count_supports(row_length):
    work_set = [item for item in dataset]
    for x in possibilities:
        sup_list = []
        for y in work_set:
            should_add = True
            for i in range(0, row_length):
                if x[i] != "":
                    if y[i] != x[i]:
                        should_add = False
                        break
            if should_add:
                sup_list.append(y)
        #print(len(sup_list))
        supports = len(sup_list)/len(work_set)
        x.append(supports)
        if supports == 0:
            x.append(0.0)
            continue
        yes = 0
        for y in sup_list:
            if y[4] == "Yes":
                yes += 1
        #print(supports)
        #print(yes)
        x.append(yes/len(sup_list))

def clear_irrelevant(pos_to_check):
    cleared = []
    for i in possibilities:
        if i[pos_to_check] != 0:
            cleared.append(i)
    return cleared
                
def formatted_print(to_iter):
    for x in to_iter:
        to_print = "if "
        is_first = True
        if x[4] > 0:
            for i in range(4):
                if x[i] != "":
                    if not is_first:
                        to_print+= " and "
                    to_print+= " {} = {} ".format(names[i],x[i])
                    is_first = False
            to_print+= " then play = {}".format("yes" if x[5]>0.5 else "no")
        else:
            continue
        print(to_print)

def show_res_for(params, format_output):
    rules = []
    for x in possibilities:
        should_add = True
        for i in range(4):
            if params[i] != "":
                if x[i] != params[i]:
                    should_add = False
                    break
        if should_add:
            rules.append(x)
    if format_output:
        formatted_print(rules)
    else:
        [print(x) for x in rules]
                


        
create_variations(0,[])                
count_supports(4)
#possibilities = clear_irrelevant(4)
#[print(x) for x in possibilities]
#print(len(possibilities))
#formatted_print(possibilities)

show_res_for(["","","",""],False)  
print(len(possibilities))      