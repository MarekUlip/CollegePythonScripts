def load_genome():
    genome = ""
    with open("GCA_002290365.1_ASM229036v1_genomic.fna") as genom_file:
        genom_file.readline()
        for line in genom_file:
            genome += line.rstrip()
    return genome.rstrip()


def parse_gff_file():
    indexes = []
    with open('GCA_002290365.1_ASM229036v1_genomic.gff') as gff_file:
        for line in gff_file:
            line = line.rstrip()
            values = line.split("\t")
            if len(values) > 4 and values[2] == 'gene':
                start, end = int(values[3]), int(values[4])
                indexes.append([start, end])
    return indexes


start_codon = ["ATG", "GTG", "TTG"]
end_codons = ["TAA", "TAG", "TGA"]


def find_gens(genome):
    indexes = []
    i = 0
    while i < len(genome):
        if i < len(genome) - 2:
            if genome[i:i + 3] in start_codon:
                start_index = i
                broke = False
                while True:
                    if genome[i:i + 3] in end_codons:
                        if i + 3 - start_index > 100:
                            break
                    if i > len(genome):
                        broke = True
                        break
                    # gen+=genome[i:i+3]
                    i += 3
                # gen += genome[i:i+3]
                if not broke:
                    if i + 3 - start_index > 100:
                        indexes.append([start_index, i + 3])
            else:
                i += 1
        else:
            break
    return indexes


def naive_accuracy_tester(found_indexes, known_indexes):
    true_positive = 0
    false_positive = 0
    # false_negative = 0
    for index in found_indexes:
        if index in known_indexes:
            true_positive += 1
        else:
            false_positive += 1
    print("true positive: {}\nfalse positive: {}".format(true_positive, false_positive))


complementary_map = {"A": "T", "T": "A", "C": "G", "G": "C"}


def convert_to_complementary(genome):
    complementary_genome = ""
    for unit in genome:
        complementary_genome += complementary_map[unit]
    return complementary_genome


loaded_genome = load_genome()
indexes = find_gens(loaded_genome)
indexes.extend(find_gens(convert_to_complementary(loaded_genome)[::-1]))

# analysis
print(len(indexes))
known_indexes = parse_gff_file()
unknown_starts = []
end_stuff = 0
for index in known_indexes:
    if loaded_genome[index[0]:index[0] + 3] not in start_codon:
        unknown_starts.append(loaded_genome[index[0]:index[0] + 3])
    if loaded_genome[index[0]:index[0] + 3] in end_codons:
        end_stuff += 1

print(set(unknown_starts))
print("end {}".format(end_stuff))
print(len(known_indexes))
print(len(unknown_starts))
print(len(set(unknown_starts)))
# end of analysis

# test of accuracy
naive_accuracy_tester(indexes, known_indexes)
