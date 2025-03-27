import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import os

#df = pd.read_csv("proteins_in_training.csv", index_col=0)
#list1 = list(df.index)

list1 = ['P00094', 'P0AEX9', 'P07107', 'P32081', 'P00644', 'P00750', 'P62593', 'P00099', 'P00782', 'P05121', 'P06396', 'P00720', 'P0A877', 'P42771', 'P01009', 'P00818', 'P07320', 'P19614', 'P00025', 'P0AA04', 'P24821', 'P01241', 'P81708', 'P00808', 'P04637', 'P0A2D5', 'P00974', 'P01236', 'P11540', 'P00698', 'P09850', 'Q9REI6', 'P61626', 'P37957', 'P37001', 'P11961', 'P02633', 'P0A9X9', 'P61991', 'P0AE67', 'P06654', 'P04925', 'Q08012', 'P63159', 'P08877', 'P0A910', 'P02185', 'P12931', 'P02925', 'O74035', 'P02640', 'Q9H782', 'P0A7Y4', 'P29957', 'P02751', 'P29160', 'P00004']

#df = pd.read_csv("../combined_S612.csv", index_col=0)
#list2 = list(set(df.uniprot))
# List of UniProt accession numbers
list2 = ['P60484']

# Function to fetch sequences from UniProt
def fetch_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    filename = f"{uniprot_id}.fasta"
    os.system(f"wget {url} -O {filename}")
    with open(filename) as file:
        sequence = SeqIO.read(file, "fasta").seq
    os.remove(filename)  # Remove the downloaded file after reading the sequence
    return sequence

# Fetch sequences for list1 and list2
sequences_list1 = [fetch_sequence(accession) for accession in list1]
sequences_list2 = [fetch_sequence(accession) for accession in list2]

too_close = []
# Perform sequence alignment and calculate similarity
for i, seq1 in enumerate(sequences_list1):
    for j, seq2 in enumerate(sequences_list2):
        alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
        seq1_aligned = alignment.seqA
        seq2_aligned = alignment.seqB
        similarity = sum(1 for a, b in zip(seq1_aligned, seq2_aligned) if a == b) / max(len(seq1_aligned), len(seq2_aligned))
        if similarity >= 0.25:
            too_close.append(list2[j])            
            print(f"Alignment between {list1[i]} and {list2[j]}:")
            print(f"Sequence similarity: {similarity:.2f}\n")

too_close = list(set(too_close))
print(too_close)
#with open("too_similar.txt", "w") as f:
#    for i in too_close:
#        f.write(f"{i}\n")
