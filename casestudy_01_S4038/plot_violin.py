import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
import glob as glob
#sns.set_palette("viridis")

sns.set(style="dark", palette="viridis")

path = "./"
df_full = pd.read_csv("S4038_cleaned.csv",index_col=0)

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

df_full = get_names(df_full)
df_full_correlations = get_PCC(df_full)
full_data = df_full_correlations[['method', 'PCC']]
full_data_long = pd.melt(full_data, id_vars=['method'], value_vars=['PCC'],
                        var_name='PCC', value_name='PCC_value')
full_data_long['data']=["full dataset"]*len(full_data_long)

list_of_balanced_df = glob.glob(path+"balanced_files/*.csv")
dfs_balanced = [pd.read_csv(file) for file in list_of_balanced_df]

list_of_random_df = glob.glob(path+"random_subset_files/*.csv")
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
df_combined.to_csv("PCCs.csv")
###########################################
## Make the plot
###########################################

custom_palette = {
    "balanced subset": "#d43535",   # Red for 'balanced subset'
    "random subset": "#3331a8",    # Blue for 'random subset'
    "full dataset": "black" }    # Black for 'full dataset'}

# Create a figure with desired size
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
plt.savefig("violin_plot.pdf")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_violin, x="method", y="PCC_value", alpha=0.7, hue='data', palette=custom_palette)
sns.scatterplot(data=df_scatter, x="method", y="PCC_value", color="black", s=100, zorder=10, label='full dataset', marker='o')
plt.xticks(rotation=90)
plt.ylim(0.0, 0.75)
plt.ylabel("PPC value")
plt.tight_layout()
plt.savefig("scatter_plot.pdf")
plt.show()

