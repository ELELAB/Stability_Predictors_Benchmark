import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
import requests
import os

sns.set(style="dark", palette="viridis")

path = "./"

df = pd.read_csv("../datasets/S4038_data.csv")

###############################
# step 1: CLEAN DATA
###############################
df[['AA', 'pos', 'MT']] = df['Mutation_FASTA'].str.extract(r'([A-Za-z]+)(\d+)([A-Za-z]+)')

for i in df.columns:
    if type(df[i].iloc[0])==str:
        if df[i].iloc[0].endswith(('a', 'b', 'c')):
            new_column_name = f"{i}_sequencesim"
            df[new_column_name] = df[i].str[-1]
            df[i] = df[i].str.slice(stop=-1)
            df[i] = pd.to_numeric(df[i], errors='coerce')

df['uniprot'] = df['UniProt_ID']
df['ddg'] = df['DDGexp']
#these now have their own columns
#a: The similarity of the protein in which this mutation occurs to the 
#   training set of this method is between 0-25%.
#b: The similarity of the protein in which this mutation occurs to the 
#   training set of this method is between 25-100%.
#c: This mutation exists in the training set of this method.

def get_uniprot_organism(accession_number):
    url = f"https://www.uniprot.org/uniprotkb/{accession_number}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        organism = data.get("organism", {}).get("scientificName")
        return organism
    else:
        return None

uniprot_accession = []
organism = []
for uniprot_accession_number in list(set(df.UniProt_ID)):
    print(uniprot_accession_number)
    uniprot_accession.append(uniprot_accession_number)
    organism.append(get_uniprot_organism(uniprot_accession_number))
    
organism_df = pd.DataFrame({'UniProt_ID': uniprot_accession, 'organism':organism})
organism_df['human'] = organism_df['organism'].apply(lambda x: 'human' if x == 'Homo sapiens' else 'non-human')

df = df.merge(organism_df, on="UniProt_ID")
#1YYX = denovo protein no organism. 

def get_uniprot_info(accession_number):
    url = f"https://www.uniprot.org/uniprot/{accession_number}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        entry = data.get('entry', {})
        
        gene_name = None
        if 'gene' in entry:
            gene_name = entry['gene'][0].get('name', {}).get('value')
        
        return gene_name
###############################
#STEP 2: CHECK OUT THE AMINO ACIDS:
###############################

def amino_acid_ss_dist(df):#, path):
    d = df[['AA', 'pos', 'MT']] 
    
    aa_wt_counts = d['AA'].value_counts().reset_index()
    aa_mut_counts = d['MT'].value_counts().reset_index()
    
    # Rename columns for clarity
    aa_wt_counts.columns = ['AA', 'Count_WT']
    aa_mut_counts.columns = ['AA', 'Count_MUT']
    
    # Merge the two DataFrames on the 'AA' column
    merged_counts = pd.merge(aa_wt_counts, aa_mut_counts, on='AA', how='outer').fillna(0)
    
    merged_counts['Count_MUT'] = merged_counts['Count_MUT'] *-1
    
    sns.barplot(data=merged_counts, x='AA', y='Count_WT', color="#453781FF", label='Number of WT Residues')
    sns.barplot(data=merged_counts, x='AA', y='Count_MUT', color="#2D708EFF", label='Number of mutated Residues')
    
    plt.xlabel('Amino Acid')
    plt.ylabel('Occurences')
    plt.legend()
    plt.tight_layout()
    plt.savefig("amino_acid_dist.pdf")
    plt.show()    
    return merged_counts    

merged_counts = amino_acid_ss_dist(df)

def plot_attributes(df):
    sns.set_palette("viridis")
    
    plt.figure(figsize=(15, 5))
    sns.histplot(data=df.uniprot)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("protein_dist.pdf")
    plt.show()

    plt.figure(figsize=(5, 5))
    t = sns.histplot(data=df.ddg, kde=True)
    t.set_xlabel('Experimental ΔΔG in kcal/mol')
    t.set_title("Experimental ΔΔG distribution")
    plt.tight_layout()
    plt.savefig("ddg_dist.pdf")
    plt.show()
    
plot_attributes(df)    

###############################
#STEP 3: BALANCE 
###############################
#we try to make a balance of amino acids and stability distribution
def balance_data(merged_counts, df, path, i):
    
    df_balanced = df.copy()
    df_balanced = df_balanced.reset_index(drop=True)
    
    average_amino_acids = int(abs(merged_counts.Count_MUT.mean()).round())
        
    # Calculate the occurrences of each amino acid in the 'MT' column
    amino_acid_counts = df['MT'].value_counts()
    
    # Randomly sample excess rows for each amino acid to meet the target count
    for amino_acid in amino_acid_counts.index:
        if amino_acid_counts[amino_acid] > average_amino_acids:
            rows_to_remove = np.random.choice(df_balanced[df_balanced['MT'] == amino_acid].index, size=amino_acid_counts[amino_acid] - average_amino_acids, replace=False)
            df_balanced = df_balanced.drop(rows_to_remove)
    
    df_balanced = df_balanced.reset_index(drop=True)
    
    average_gene_count = int(df_balanced['uniprot'].value_counts().reset_index()['count'].mean().round())
    gene_counts = df_balanced['uniprot'].value_counts()
    
    for uniprot in gene_counts.index:
        if gene_counts[uniprot] > average_gene_count:
            rows_to_remove = np.random.choice(df_balanced[df_balanced['uniprot'] == uniprot].index, size=min(gene_counts[uniprot] - average_gene_count, len(df_balanced[df_balanced['uniprot'] == uniprot])), replace=False)
            df_balanced = df_balanced.drop(rows_to_remove)
    
    df_balanced = df_balanced.reset_index(drop=True)
    
    n_destabilizing = len(df_balanced[df_balanced.ddg > 0])
    n_stabilizing = len(df_balanced[df_balanced.ddg <= 0])
    
    difference = n_destabilizing-n_stabilizing
    
    rows_to_remove = np.random.choice(df_balanced[df_balanced['ddg'] > 0].index, size=difference, replace=False)
    df_balanced = df_balanced.drop(rows_to_remove)
    
    df_balanced.to_csv(path+f"{i+1}_S{len(df_balanced)}.csv")
    
    length = len(df_balanced)
    
    return df_balanced, length

################# 
# STEP 4: RANDOM SUBSET
#################
def random_subset(df, path, i, average):
    df_subset = df.copy().reset_index(drop=True)
    
    rows_to_keep = np.random.choice(df_subset.index, size=average, replace=False)
    
    df_subset = df_subset.loc[rows_to_keep].reset_index(drop=True)
    
    df_subset.to_csv(path + f"{i+1}_S{average}.csv", index=False)
    
    return df_subset

################# 
# STEP 5: CREATE DATA
#################

df.to_csv("S4038_cleaned.csv")

#if not os.path.exists(path+"balanced_files/"):
#    os.makedirs(path+"balanced_files/")

#if not os.path.exists(path+"random_subset_files/"):
#    os.makedirs(path+"random_subset_files/")

#length_list = []
#for i in range(10):
#    df_balanced, length = balance_data(merged_counts, df, path+"balanced_files/", i)
#    length_list.append(length)
    
#average = int(np.array(length_list).mean().round())
#print(f"THIS IS THE AVERAGE {average}")

#for i in range(10):
#    df_subset = random_subset(df, path+"random_subset_files/", i, average)
