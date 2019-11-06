def load_genome():
    genome = ""
    with open("genom_nausica.fna") as genom_file:
        genom_file.readline()
        for line in genom_file:
            genome+=line.rstrip()
    return genome.rstrip()

#print(load_genome())
loaded_genome = load_genome()
test= ["A","T","C","G"]
complementary_map = {"A":"T", "T":"A","C":"G", "G":"C"}
def convert_to_complementary(genome):
    complementary_genome = ""
    for unit in genome:
        complementary_genome += complementary_map[unit]
    return complementary_genome

convert_to_complementary(loaded_genome)

start_codon = "ATG"
end_codons = ["TAA", "TAG", "TGA"]
def find_gens(genome):
    gens = []
    for i in range(len(genome)):
        if genome[i] == start_codon[0] and i < len(genome)-2:
            if genome[i+1] == start_codon[1]:
                if genome[i+2] == start_codon[2]:
                    gen = start_codon
                    while genome[i:i+3] not in end_codons:
                        i+=3
                        gen+=genome[i:i+3]
                    gens.append(gen)
    return gens

def find_gens_with_ignore(genome):
    gens = []
    i = 0
    while i < len(genome):
        if i < len(genome)-2:
            if genome[i:i+3] == start_codon:
                gen = ""
                while genome[i:i+3] not in end_codons:
                    gen+=genome[i:i+3]
                    i+=3
                gen += genome[i:i+3]
                gens.append(gen)
            else:
                i+=1
        else:
            break
    return gens


amino_map = dict.fromkeys(["TTT","TTC"],"Phe")
amino_map.update(dict.fromkeys(["TTA","TTG","CTT","CTC","CTA","CTG"],"Leu"))
amino_map.update(dict.fromkeys(["ATT","ATC","ATA"],"Ile"))
gencode = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'}
gencode_remapper = {"I":"Ile","M":"Met","T":"Thr","N":"Asn","K":"Lys","S":"Ser","R":"Arg","L":"Leu","P":"Pro","H":"His","Q":"Gln","V":"Val","A":"Ala","D":"Asp","E":"Glu","G":"Gly","F":"Phe","Y":"Tyr","C":"Cys","W":"Trp","_":"STOP"}
def convert_to_amino(gen):
    amino = ""
    for i in range(0,len(gen),3):
        amino += gencode_remapper[gencode[gen[i:i+3]]] +"-"
    print(amino[:-1])
    return amino[:-1]

print(len(loaded_genome))
print(loaded_genome[:200])
found_gens = find_gens_with_ignore(convert_to_complementary(loaded_genome))#find_gens(convert_to_complementary(loaded_genome))
#print(found_gens)
print(len(found_gens))
found_gens = find_gens_with_ignore(loaded_genome)#find_gens(convert_to_complementary(loaded_genome))
#print(found_gens)
print(len(found_gens))
convert_to_amino(found_gens[0])