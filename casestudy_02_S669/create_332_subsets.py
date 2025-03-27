#balance S615 (S612)

import pandas as pd
import numpy as np
import os

path = "./"

df = pd.read_csv(path+"combined_S612_thermompnn25.csv", index_col=0)
df = df[df['ThermoMPNN>=25%sim'] == False]

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
    
    return merged_counts    

merged_counts = amino_acid_ss_dist(df)

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
    
if not os.path.exists(path+"balanced_files_332/"):
    os.makedirs(path+"balanced_files_332/")

if not os.path.exists(path+"random_subset_files_332/"):
    os.makedirs(path+"random_subset_files_332/")

length_list = []
for i in range(10):
    df_balanced, length = balance_data(merged_counts, df, path+"balanced_files_332/", i)
    length_list.append(length)
    
average = int(np.array(length_list).mean().round())
print(f"THIS IS THE AVERAGE {average}")

for i in range(10):
    df_subset = random_subset(df, path+"random_subset_files_332/", i, average)
