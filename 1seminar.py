import copy

def create_variations3():
    rules = []
    for i in range(6):
        for j in range(i+1,6):
            for k in range(j+1,6):
                rules.append([i,j,k])
    print(rules)

def count_support(item, datas):
    support = 0
    for itemset in datas:
        if all(elem in itemset for elem in item):
            support += 1
    #print("{}: {}".format(item,support/len(datas)))
    return support/len(datas)

def count_conf(item, Y,datas):
    conf = 0
    X_count = 0
    for itemset in datas:
        if all(elem in itemset for elem in item):
            X_count += 1
            if Y in itemset: 
                conf += 1
    #print("{}: {}".format(item,support/len(datas)))
    return conf/X_count

rules = []
def create_rules(rule_len, num_of_possibilities, exclusions = None):
    for i in range(1,num_of_possibilities-rule_len+1):
        if exclusions is not None and i in exclusions:
            continue
        create_rule(i+1,[i],rule_len,num_of_possibilities)
    print(rules)
    return rules

def create_rule(num, field, rule_len, max_num, exclusions = None):
    if len(field) == rule_len:
        #supp = count_support(field,dataset)
        rules.append(field)
        return
    
    for i in range(num,max_num):
        if exclusions is not None and i in exclusions:
            continue
        f = copy.deepcopy(field)
        f.append(i)
        create_rule(i+1,f,rule_len,max_num)

def create_rules_a(start_from, field_for_rule, rule_len, create_till, created_rules, transactions, minsup):
    if len(field_for_rule) == rule_len:
        supp = count_support(field_for_rule,transactions)
        if supp >= minsup:
            print("{}: {}".format(field_for_rule,supp))
            created_rules.append(field_for_rule)
        return
    
    for i in range(start_from,create_till):
        temp_field = copy.deepcopy(field_for_rule)
        temp_field.append(i)
        create_rules_a(i+1,temp_field,rule_len,create_till,created_rules,transactions,minsup)

def apriori(transactions, minsup):
    k = 0
    rules_to_create = []
    create_rules_a(1,[],1,75,rules_to_create,transactions,minsup)
    F = [rules_to_create]
    while len(F[k]) > 0:
        rules_to_create = []
        for rule in F[k]:
            create_rules_a(rule[len(rule)-1]+1,rule,k+2,75,rules_to_create,transactions,minsup)
        F.append(rules_to_create)
        k += 1
    input()
    return F
    


def do_final_things(F, transactions, minsup):
    rule_sum = 0
    min_conf = 1
    conf_rules = []
    for rules in F:
        rule_sum+=len(rules)
        for rule in rules:
            if len(rule) <= 1:
                continue
            for j in range(len(rule)):
                temp_rule = []
                Y = 0
                for i in range(len(rule)):
                    if i == j:
                        Y = rule[i]
                    else:
                        temp_rule.append(rule[i])
                """temp_rule = rule[:-1]
                Y = rule[len(rule)-1]"""
                conf = count_conf(temp_rule,Y,transactions)
                print("{}  -> {} : {}".format(temp_rule,Y,conf))
                conf_rules.append([temp_rule,Y])
                if conf < min_conf:
                    min_conf = conf
    print("Min confidence is {}".format(min_conf))
    print("Number of freq item sets that have min support {} is {}".format(minsup, rule_sum))
    print("Number of rules that have min confidence {} is {}".format(min_conf, len(conf_rules)))
    return conf_rules
def load_dataset(file_path):
    dataset = []
    with open(file_path) as dset:
        for line in dset:
            new_row = line.split(" ")
            dataset.append([int(item) for item in new_row[:len(new_row)-1]])
    return dataset


create_rules(3,6)
input()
dataset = load_dataset("chess.dat")
print(len(do_final_things(apriori(dataset,0.950),dataset,0.950)))
#create_variations3()
#for i in range(5):
#    create_rules(i+1,75)
#print(len(load_dataset("chess.dat")))
