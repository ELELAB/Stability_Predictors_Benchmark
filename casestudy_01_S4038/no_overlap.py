import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
import os
import glob as glob
from scipy.stats import ttest_1samp, ttest_ind

sns.set(style="dark", palette="viridis")

path = "./"
df = pd.read_csv("S4038_cleaned.csv", index_col=0)

#a: The similarity of the protein in which this mutation occurs to the 
#   training set of this method is between 0-25%.
#b: The similarity of the protein in which this mutation occurs to the 
#   training set of this method is between 25-100%.
#c: This mutation exists in the training set of this method.

sequence_sim_columns = [col for col in df.columns if col.endswith('_sequencesim')]
df_filtered = df[~df[sequence_sim_columns].isin(['b', 'c']).any(axis=1)]

df_filtered.to_csv("S568.csv")

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
    plt.savefig("amino_acid_dist_S568.pdf")
    plt.show()    
    return merged_counts    

merged_counts = amino_acid_ss_dist(df_filtered)

def plot_attributes(df):
    sns.set_palette("viridis")
    
    plt.figure(figsize=(15, 5))
    sns.histplot(data=df.uniprot)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("protein_dist_S568.pdf")
    plt.show()

    plt.figure(figsize=(5, 5))
    t = sns.histplot(data=df.ddg, kde=True)
    t.set_xlabel('Experimental ΔΔG in kcal/mol')
    t.set_title("Experimental ΔΔG distribution")
    plt.tight_layout()
    plt.savefig("ddg_dist_S568.pdf")
    plt.show()

plot_attributes(df_filtered)

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

if not os.path.exists(path+"balanced_files_S568/"):
    os.makedirs(path+"balanced_files_S568/")

if not os.path.exists(path+"random_subset_files_S568/"):
    os.makedirs(path+"random_subset_files_S568/")

length_list = []
for i in range(10):
    df_balanced, length = balance_data(merged_counts, df_filtered, path+"balanced_files_S568/", i)
    length_list.append(length)
    
average = int(np.array(length_list).mean().round())
print(f"THIS IS THE AVERAGE {average}")

for i in range(10):
    df_subset = random_subset(df_filtered, path+"random_subset_files_S568/", i, average)

################# 
# STEP 6: GET PCC
#################

def get_names(df):
    df = df[['DDGexp', 'INPS', 'MUpro', 'DynaMut2',
           'I-Mutant2.0', 'I-Mutant2.0-Seq', 'PoPMuSiC_2.1', 'ACDC-NN',
           'SAAFEC-SEQ', 'MAESTRO', 'ThermoNet', 'PremPS', 'ACDC-NN-Seq',
           'BayeStab', 'FoldX', 'AUTO-MUTE(RF)', 'AUTO-MUTE(SVM)', 'INPS3D',
           'DDMut', 'SimBa-IB', 'SimBa-SYM', 'ThermoMPNN', 'RaSP', 'DDGun3D',
           'MultiMutate', 'Rosetta', 'DDGun', 'CUPSAT']]
    
    df = df.rename(columns={"DDGexp": "ddg"})
    return df

def get_PCC(df): 
    variable = []
    pcc_values = []
    scc_values = []
    mae_values = []
    
    # Iterate over each column (excluding 'ddg')
    for i, column in enumerate(df.columns[1:]):     
        print(column)
        variable.append(column)
        
        cleaned_df = df[['ddg', column]].dropna()
        cleaned_df['ddg'] = cleaned_df['ddg'].astype(float)
        cleaned_df[column] = cleaned_df[column].astype(float)
        
        pearson_corr, _ = stats.pearsonr(cleaned_df['ddg'], cleaned_df[column])
        pcc_values.append(pearson_corr)
        spearman_corr, _ = stats.spearmanr(cleaned_df['ddg'], cleaned_df[column])
        scc_values.append(spearman_corr)
        mae = mean_absolute_error(cleaned_df['ddg'], cleaned_df[column])
        mae_values.append(mae)
    
    df_correlation = pd.DataFrame({'method':  variable, 
                                   'PCC': pcc_values,
                                   'SCC': scc_values,
                                   'MAE': mae_values})
    return df_correlation


df_full = df_filtered
df_full = get_names(df_full)
df_full_correlations = get_PCC(df_full)
full_data = df_full_correlations[['method', 'PCC']]
full_data_long = pd.melt(full_data, id_vars=['method'], value_vars=['PCC'], var_name='PCC', value_name='PCC_value')
full_data_long['data']=["full dataset"]*len(full_data_long)

list_of_balanced_df = glob.glob(path+"balanced_files_S568/*.csv")
dfs_balanced = [pd.read_csv(file) for file in list_of_balanced_df]

list_of_random_df = glob.glob(path+"random_subset_files_S568/*.csv")
dfs_random = [pd.read_csv(file) for file in list_of_random_df]

def collect_all(dfs):
    results = []
    for i, df in enumerate(dfs, start=1):
        df = get_names(df)
        d = get_PCC(df)
        # Retain only the 'PCC' and 'method' columns and round 'PCC' to 2 decimals
        d = d[['PCC', 'method']]
        d['PCC'] = d['PCC'].round(2)
        # Rename 'PCC' column
        d = d.rename(columns={'PCC': f'PCC_{i}'})
        # Append the result to the list
        results.append(d)
    # Merge all results on the 'method' column
    final_result = results[0]
    for d in results[1:]:
        final_result = final_result.merge(d, on='method', how='outer')
    final_result=final_result[['method','PCC_1','PCC_2', 'PCC_3', 'PCC_4', 'PCC_5', 'PCC_6', 'PCC_7','PCC_8', 'PCC_9', 'PCC_10']]
    means=[]
    for i in range(len(final_result)):
       means.append(final_result.iloc[i][1:].mean())
    final_result['mean']=means
    final_result=final_result.sort_values(by="mean", ascending=False)
    boxplot_df = final_result.set_index("method").T
    sns.boxplot(boxplot_df)
    plt.tick_params(axis='x', rotation=90)
    balanced=final_result.sort_values(by="method")
    return balanced

balanced_pcc = collect_all(dfs_balanced)
balanced_long = pd.melt(balanced_pcc, id_vars=['method'], value_vars=['PCC_1', 'PCC_2', 'PCC_3', 'PCC_4', 'PCC_5', 'PCC_6', 'PCC_7', 'PCC_8', 'PCC_9', 'PCC_10'],
                        var_name='PCC', value_name='PCC_value')
balanced_long['data']=["balanced subset"]*len(balanced_long)

random_pcc = collect_all(dfs_random)

random_long = pd.melt(random_pcc, id_vars=['method'], value_vars=['PCC_1', 'PCC_2', 'PCC_3', 'PCC_4', 'PCC_5', 'PCC_6', 'PCC_7', 'PCC_8', 'PCC_9', 'PCC_10'],
                        var_name='PCC', value_name='PCC_value')
random_long['data']=["random subset"]*len(random_long)

df_combined = pd.concat([full_data_long, balanced_long, random_long])
df_combined.to_csv("PCCs_S568.csv")
###########################################
## Make the plot
###########################################

custom_palette = {
     "balanced subset": "#d43535",   # Red for 'balanced subset'
     "random subset": "#3331a8",    # Blue for 'random subset'
     "full dataset": "black" }    # Black for 'full dataset'}

# # Create a figure with desired size
plt.figure(figsize=(10, 6))

# Separate the 'full dataset' from the rest
df_violin = df_combined[df_combined['data'] != 'full dataset'] 
df_violin = df_violin.sort_values(by="method")
df_scatter = df_combined[df_combined['data'] == 'full dataset']
df_scatter = df_scatter.sort_values(by="method")

sns.violinplot(data=df_violin, x="method", y="PCC_value", alpha=0.7, hue='data', palette=custom_palette)
sns.scatterplot(data=df_scatter, x="method", y="PCC_value", color="black", s=100, zorder=10, label='full dataset', marker='o')
plt.xticks(rotation=90)
plt.ylim(0.0, 0.75)
plt.ylabel("PPC value")
plt.tight_layout()
plt.savefig("violin_plot_S568.pdf")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_violin, x="method", y="PCC_value", alpha=0.7, hue='data', palette=custom_palette)
sns.scatterplot(data=df_scatter, x="method", y="PCC_value", color="black", s=100, zorder=10, label='full dataset', marker='o')
plt.xticks(rotation=90)
plt.ylim(0.0, 0.75)
plt.ylabel("PPC value")
plt.tight_layout()
plt.savefig("scatter_plot_S568.pdf")
plt.show()

def get_pvalues(df):
    random_subset = df[df['data'] == 'random subset']
    full_dataset = df[df['data'] == 'full dataset']
    balanced_subset = df[df['data'] == 'balanced subset']
    method_list = []
    random_results_full = []
    balanced_results_full = []
    compare_results_full = []
    for method in df['method'].unique():
        method_list.append(method)
        random_method_values = random_subset[random_subset['method'] == method]['PCC_value']
        full_method_values = full_dataset[full_dataset['method'] == method]['PCC_value']
        single_value_full = full_method_values.iloc[0]
        balanced_method_values = balanced_subset[balanced_subset['method'] == method]['PCC_value']
        
        #One-sample t-test for random_method_values vs. full_method_value
        #This test checks if the mean of each sample (random_method_values or balanced_method_values) 
        #is significantly different from the single PCC_value in full_method_values.
        stat_random, pval_random = ttest_1samp(random_method_values, single_value_full)
        random_results_full.append(pval_random)
    
        #One-sample t-test for balanced_method_values vs. full_method_value
        stat_balanced, pval_balanced = ttest_1samp(balanced_method_values, single_value_full)
        balanced_results_full.append(pval_balanced)
        
        #Independent t-test between random_method_values and balanced_method_values
        stat_compare, pval_compare = ttest_ind(random_method_values, balanced_method_values, equal_var=False)
        compare_results_full.append(pval_compare)
    
    df_pval = pd.DataFrame({'method': method_list, 
                            'random_like_full': random_results_full,
                            'balanced_like_full': balanced_results_full,
                            'random_balanced_identical': compare_results_full})
    return df_pval


df_pval = get_pvalues(df_combined)
df_pval.to_csv("pvalues_S568.csv")
