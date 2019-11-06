def load_genome():
    genome = ""
    with open("GCA_002290365.1_ASM229036v1_genomic.fna") as genom_file:
        genom_file.readline()
        for line in genom_file:
            genome+=line.rstrip()
    return genome.rstrip()

def parse_gff_file():
    indexes = []
    #lengths = []
    with open('GCA_002290365.1_ASM229036v1_genomic.gff') as gff_file:
        for line in gff_file:
            line = line.rstrip()
            values = line.split("\t")
            if len(values) > 4 and values[2] == 'gene':
                start, end = int(values[3]), int(values[4])
                indexes.append([start,end])
      #          length = end - start + 1
     #           lengths.append(length)
                #if length >= 1000:
                    #print(line)
    #print(min(lengths))
    return indexes
start_codon = ["ATG","GTG","TTG"]
end_codons = ["TAA", "TAG", "TGA"]

def find_gens_with_ignore(genome):
    #gens = []
    indexes = []
    i = 0
    while i < len(genome):
        if i < len(genome)-2:
            if genome[i:i+3] in start_codon:
                start_index = i
                #gen = ""
                broke = False
                while True:
                    if genome[i:i+3]  in end_codons:
                        if i+3 - start_index > 70:
                            break
                    if i > len(genome):
                        broke = True
                        break
                    #gen+=genome[i:i+3]
                    i+=3
                #gen += genome[i:i+3]
                if not broke:
                    if i+3 - start_index > 70:
                        indexes.append([start_index,i+3])
            else:
                i+=1
        else:
            break
    return indexes

def naive_accuracy_tester(found_indexes, known_indexes):
    for index in known_indexes:
        if index in found_indexes:
            print('GOTCHA!')

"""def find_gens_with_ignore_complementary(genome):
    indexes = []
    i = len(genome)-1
    while i > 0:
        if i > 0 +2:
            if genome[]"""

loaded_genome = load_genome()
indexes = find_gens_with_ignore(loaded_genome)
indexes.extend(find_gens_with_ignore(loaded_genome[::-1]))
#indexes = find_gens_with_ignore(loaded_genome[::-1])
print(len(indexes))
#print(indexes)

print(loaded_genome[2900:3550])
known_indexes = parse_gff_file()
unknown_starts = []
end_stuff = 0
for index in known_indexes:
    if loaded_genome[index[0]:index[0]+3] not in start_codon:
        #print(loaded_genome[index[0]:index[0]+3])
        unknown_starts.append(loaded_genome[index[0]:index[0]+3])
    if loaded_genome[index[0]:index[0]+3] in end_codons:
        print('WTF?')
        end_stuff += 1
    
print(set(unknown_starts))
print("end {}".format(end_stuff))
print(len(known_indexes))
print(len(unknown_starts))
print(len(set(unknown_starts)))

naive_accuracy_tester(indexes,known_indexes)