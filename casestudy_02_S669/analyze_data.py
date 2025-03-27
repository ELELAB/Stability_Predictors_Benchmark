import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error
from itertools import combinations
from sklearn.linear_model import LinearRegression
import numpy as np
sns.set(style="dark", palette="viridis")
 
def make_data_dist_figure(df):
    
    sns.set(style="dark", palette="viridis")
    #fig, axes = plt.subplots(2, 2, figsize=(12, 12))#, sharey=True) 
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'width_ratios': [1, 3]})
    
    # ax[0]
    sns.histplot(data=df.ddg, kde=True, ax=axes[0,0])
    axes[0,0].set_xlabel('Experimental ΔΔG in kcal/mol')
    axes[0,0].set_title("Experimental ΔΔG distribution")
    
    sns.histplot(df, x='gene', hue='human', multiple='stack', shrink=0.8, ax=axes[0,1])
    axes[0,1].tick_params(axis='x', rotation=90)  # Rotate x-axis labels for axes[1]
    axes[0,1].set_title('Distribution of genes with human/non-human')
    
    d = df[['AA_WT', 'pos', 'AA_MUT']] 
    
    aa_wt_counts = d['AA_WT'].value_counts().reset_index()
    aa_mut_counts = d['AA_MUT'].value_counts().reset_index()
    
    # Rename columns for clarity
    aa_wt_counts.columns = ['AA', 'Count_WT']
    aa_mut_counts.columns = ['AA', 'Count_MUT']
    
    # Merge the two DataFrames on the 'AA' column
    merged_counts = pd.merge(aa_wt_counts, aa_mut_counts, on='AA', how='outer').fillna(0)
    
    merged_counts['Count_MUT'] = merged_counts['Count_MUT'] *-1
    
    sns.barplot(data=merged_counts, x='AA', y='Count_WT', color="#3b528b", label='Number of WT Residues', ax=axes[1,0])
    sns.barplot(data=merged_counts, x='AA', y='Count_MUT', color="#21918c", label='Number of mutated Residues', ax=axes[1,0])
    ax=axes[1,0].set_xlabel('Amino Acid')
    ax=axes[1,0].set_ylabel('Occurences')
    ax=axes[1,0].legend()
    
    sns.boxplot(x=df.ddg, y=df.method, color="lightgrey", ax=axes[1, 1])
    sns.scatterplot(x=df.ddg, y=df.method, hue=df.measure, palette="viridis", ax=axes[1, 1], edgecolor=None)
    #scatter_plot.legend(loc='upper left', bbox_to_anchor=(1, 1))
      
    for i, ax in enumerate(axes.flatten()):
        panel_label = chr(ord('A') + i)  # Convert index to corresponding letter
        ax.text(-0.06, 1.06, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.savefig('distribution.pdf')
    plt.show()
    
    return

def make_prediction_figure(df):
    
    method = []
    pcc_list = []
    scc_list = []
    mae_list = []
    
    sns.set(style="dark", palette="viridis")
    fig, axes = plt.subplots(3, 2, figsize=(11, 15))#, sharey=True) 
    
    df_temp = df[["ddg", "thermompnn"]].dropna()
    
    sns.scatterplot(data=df_temp, x='ddg', y='thermompnn', ax=axes[0, 0], alpha=0.5, edgecolor=None)
    axes[0, 0].set_xlabel('Experimental ΔΔG in kcal/mol')
    axes[0, 0].set_ylabel('Predicted ΔΔG in kcal/mol')
    axes[0, 0].plot([-2, 9], [-2, 9], '--', color='black', linewidth=1, label='Perfect Correlation')
    axes[0, 0].set_ylim(-4, 10)
    pearson_corr, _ = stats.pearsonr(df_temp['ddg'], df_temp['thermompnn'])
    spearman_corr, _ = stats.spearmanr(df_temp['ddg'], df_temp['thermompnn'])
    mae = mean_absolute_error(df_temp['ddg'], df_temp['thermompnn'])
    method.append("thermoMPNN")
    pcc_list.append(pearson_corr)
    scc_list.append(spearman_corr)
    mae_list.append(mae)
    print(f"thermoMPNN, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    axes[0, 0].set_title(f"thermoMPNN, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    
    #df_ddgunseq = df[['ddg', 'ddgun_seq']]
    #df_ddgunseq = df_ddgunseq[df_ddgunseq.apply(lambda row: 'NA - missing file' not in row.values, axis=1)]
    #df_ddgunseq['ddgun_seq'] = df_ddgunseq['ddgun_seq'].astype(float)
    df_temp = df[["ddg", "ddgun_seq"]].dropna()
    sns.scatterplot(data=df, x='ddg', y='ddgun_seq', ax=axes[0, 1], alpha=0.5, edgecolor=None)
    axes[0, 1].set_xlabel('Experimental ΔΔG in kcal/mol')
    axes[0, 1].set_ylabel('Predicted ΔΔG in kcal/mol')
    axes[0, 1].plot([-2, 9], [-2, 9], '--', color='black', linewidth=1, label='Perfect Correlation')
    axes[0, 1].set_ylim(-4, 10)
    pearson_corr, _ = stats.pearsonr(df_temp['ddg'], df_temp['ddgun_seq'])
    spearman_corr, _ = stats.spearmanr(df_temp['ddg'], df_temp['ddgun_seq'])
    mae = mean_absolute_error(df_temp['ddg'], df_temp['ddgun_seq'])
    method.append("ddGun seq")
    pcc_list.append(pearson_corr)
    scc_list.append(spearman_corr)
    mae_list.append(mae)
    print(f"ddGun, Sequence, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    axes[0, 1].set_title(f"ddGun, Sequence, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    
    df_temp = df[["ddg", "ddgun_3d"]].dropna()
    sns.scatterplot(data=df, x='ddg', y='ddgun_3d', ax=axes[1, 0], alpha=0.5, edgecolor=None)
    axes[1, 0].set_xlabel('Experimental ΔΔG in kcal/mol')
    axes[1, 0].set_ylabel('Predicted ΔΔG in kcal/mol')
    axes[1, 0].plot([-2, 9], [-2, 9], '--', color='black', linewidth=1, label='Perfect Correlation')
    axes[1, 0].set_ylim(-4, 10)
    pearson_corr, _ = stats.pearsonr(df_temp['ddg'], df_temp['ddgun_3d'])
    spearman_corr, _ = stats.spearmanr(df_temp['ddg'], df_temp['ddgun_3d'])
    mae = mean_absolute_error(df_temp['ddg'], df_temp['ddgun_3d'])
    method.append("ddGun 3d")
    pcc_list.append(pearson_corr)
    scc_list.append(spearman_corr)
    mae_list.append(mae)
    print(f"ddGun, 3D, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    axes[1, 0].set_title(f"ddGun, 3D, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    
    
    df_m = df[['ddg', 'mutatex']]
    df_m = df_m[df_m.apply(lambda row: 'NA - file missing' not in row.values, axis=1)]
    df_m['mutatex'] = df_m['mutatex'].astype(float)
    df_m = df_m.dropna()
    
    sns.scatterplot(data=df, x='ddg', y='mutatex', ax=axes[1, 1], alpha=0.5, edgecolor=None)
    axes[1, 1].set_xlabel('Experimental ΔΔG in kcal/mol')
    axes[1, 1].set_ylabel('Predicted ΔΔG in kcal/mol')
    axes[1, 1].plot([-2, 9], [-2, 9], '--', color='black', linewidth=1, label='Perfect Correlation')
    axes[1, 1].set_ylim(-4, 10)
    pearson_corr, _ = stats.pearsonr(df_m['ddg'], df_m['mutatex'])
    spearman_corr, _ = stats.spearmanr(df_m['ddg'], df_m['mutatex'])
    mae = mean_absolute_error(df_m['ddg'], df_m['mutatex'])
    method.append("MutateX")
    pcc_list.append(pearson_corr)
    scc_list.append(spearman_corr)
    mae_list.append(mae)
    print(f"MutateX, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    axes[1, 1].set_title(f"MutateX, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    
    df_ro = df[['ddg', 'rosetta']]
    df_ro = df_ro[df_ro.apply(lambda row: 'NA - file missing' not in row.values, axis=1)]
    df_ro['rosetta'] = df_ro['rosetta'].astype(float)
    df_ro = df_ro.dropna()
    sns.scatterplot(data=df, x='ddg', y='rosetta', ax=axes[2, 0], alpha=0.5, edgecolor=None)
    axes[2, 0].set_xlabel('Experimental ΔΔG in kcal/mol')
    axes[2, 0].set_ylabel('Predicted ΔΔG in kcal/mol')
    axes[2, 0].plot([-2, 9], [-2, 9], '--', color='black', linewidth=1, label='Perfect Correlation')
    axes[2, 0].set_ylim(-4, 10)
    pearson_corr, _ = stats.pearsonr(df_ro['ddg'], df_ro['rosetta'])
    spearman_corr, _ = stats.spearmanr(df_ro['ddg'], df_ro['rosetta'])
    mae = mean_absolute_error(df_ro['ddg'], df_ro['rosetta'])
    method.append("Rosetta")
    pcc_list.append(pearson_corr)
    scc_list.append(spearman_corr)
    mae_list.append(mae)
    print(f"Rosetta ref2015 cartesian, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    axes[2, 0].set_title(f"Rosetta ref2015 cartesian, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    
    df_ra = df[['ddg', 'rasp']]
    df_ra = df_ra[df_ra.apply(lambda row: 'NA - file missing' not in row.values, axis=1)]
    df_ra['rasp'] = df_ra['rasp'].astype(float)
    df_ra = df_ra.dropna()
    sns.scatterplot(data=df, x='ddg', y='rasp', ax=axes[2, 1], alpha=0.5, edgecolor=None)
    axes[2, 1].set_xlabel('Experimental ΔΔG in kcal/mol')
    axes[2, 1].set_ylabel('Predicted ΔΔG in kcal/mol')
    axes[2, 1].plot([-2, 9], [-2, 9], '--', color='black', linewidth=1, label='Perfect Correlation')
    axes[2, 1].set_ylim(-4, 10)
    pearson_corr, _ = stats.pearsonr(df_ra['ddg'], df_ra['rasp'])
    spearman_corr, _ = stats.spearmanr(df_ra['ddg'], df_ra['rasp'])
    mae = mean_absolute_error(df_ra['ddg'], df_ra['rasp'])
    method.append("RaSP")
    pcc_list.append(pearson_corr)
    scc_list.append(spearman_corr)
    mae_list.append(mae)
    print(f"RaSP, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    axes[2, 1].set_title(f"RaSP, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
    
    for i, ax in enumerate(axes.flatten()):
        panel_label = chr(ord('A') + i)  # Convert index to corresponding letter
        ax.text(-0.06, 1.06, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.savefig('core_tools_performance.pdf')
    plt.show()
    
    df_performance = pd.DataFrame({'method': method, 'PCC': pcc_list, 'SCC': scc_list, 'MAE':mae_list})
    
    return df_performance

def multiple_linear_correlations_part1(df, path):
    
    df1 = df.copy()
    parameters = ['mutatex', 'rosetta', 'rasp', 'ddgun_seq', 'ddgun_3d']
    
    model_identifier = []
    variables = []
    linear_combination = []
        
    all_combinations = []
    for r in range(1, len(parameters) + 1):
        combinations_r = combinations(parameters, r)
        all_combinations.extend(combinations_r)
    
    for i, combo in enumerate(all_combinations):
        X = df1[list(combo)]  # Independent variables
        y = df1['ddg']  # Dependent variable
        # Fit the model
        model = LinearRegression().fit(X, y)
        df1[f'predicted_{i+1}'] = model.predict(X)
        model_identifier.append(f"Model {i+1}")
        parameter_combo = list(combo)
        variables.append(f"{', '.join(parameter_combo)}")
        coef = model.coef_.round(2)
        coeficient_list = []
        for i in range(len(parameter_combo)): 
            coeficient_list.append(f"{coef[i]}*{parameter_combo[i]}")
        combined_coefs = ""
        for coef in coeficient_list:
            if coef[0] != "-":
                combined_coefs += "+" + coef
            else:
                combined_coefs += coef[0] + coef[1:]
        # Remove the leading "+" sign if present
        combined_coefs = combined_coefs.lstrip("+")
        linear_combination.append(f"ddG = {model.intercept_.round(2)}+{combined_coefs}")
    
    df_coef = pd.DataFrame({'model': model_identifier, 'method': variables, 'MLR':linear_combination}) 
    return df1, all_combinations, df_coef

def multiple_linear_correlations_part2(df, path, number):    
    df2 = df.dropna(subset='thermompnn')
    number = number+1
    parameters = ['mutatex', 'rosetta', 'rasp', 'ddgun_seq', 'ddgun_3d', 'thermompnn']
    
    model_identifier = []
    variables = []
    linear_combination = []
        
    all_combinations = []
    for r in range(1, len(parameters) + 1):
        combinations_r = combinations(parameters, r)
        all_combinations.extend(combinations_r)
        
    corrected = []  
    for item in all_combinations:
        if 'thermompnn' in item: 
            corrected.append(item)
    
    for i, combo in enumerate(corrected):
        X = df2[list(combo)]  # Independent variables
        y = df2['ddg']  # Dependent variable
        # Fit the model
        model = LinearRegression().fit(X, y)
        df2[f'predicted_{i+number}'] = model.predict(X)
        model_identifier.append(f"Model {i+number}")
        parameter_combo = list(combo)
        variables.append(f"{', '.join(parameter_combo)}")
        coef = model.coef_.round(2)
        coeficient_list = []
        for i in range(len(parameter_combo)): 
            coeficient_list.append(f"{coef[i]}*{parameter_combo[i]}")
        combined_coefs = ""
        for coef in coeficient_list:
            if coef[0] != "-":
                combined_coefs += "+" + coef
            else:
                combined_coefs += coef[0] + coef[1:]
        # Remove the leading "+" sign if present
        combined_coefs = combined_coefs.lstrip("+")
        linear_combination.append(f"ddG = {model.intercept_.round(2)}+{combined_coefs}")
    
    df_coef = pd.DataFrame({'model': model_identifier, 'method': variables, 'MLR':linear_combination}) 
    return df2, corrected, df_coef

def plot_scatter_grid(df, all_combinations):
    
    method = []
    pcc_list = []
    scc_list = []
    mae_list = []
    
    fig, axes = plt.subplots(8, 8, figsize=(40, 40))

    for i, ax in enumerate(axes.flatten()):
        if i < len(all_combinations):  # Ensure index is within the range of all_combinations
            method.append(", ".join(all_combinations[i]))
            model_column = f'predicted_{i+1}'
            sns.scatterplot(data=df, x='ddg', y=model_column, ax=ax, alpha=0.5, edgecolor=None)
            ax.set_xlabel('Experimental ΔΔG in kcal/mol')
            ax.set_ylabel('Predicted ΔΔG in kcal/mol')
            ax.plot([-2, 9], [-2, 9], '--', color='black', linewidth=1, label='Perfect Correlation')
            ax.set_ylim(-4, 10)
            df_temp = df[['ddg', model_column]].dropna()
            pearson_corr, _ = stats.pearsonr(df_temp['ddg'], df_temp[model_column])
            pcc_list.append(pearson_corr)
            spearman_corr, _ = stats.spearmanr(df_temp['ddg'], df_temp[model_column])
            scc_list.append(spearman_corr)
            mae = mean_absolute_error(df_temp['ddg'], df_temp[model_column])
            mae_list.append(mae)
            print(f"Model {i+1}, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")
            ax.set_title(f"{model_column}, PCC={pearson_corr:.2f}, SCC={spearman_corr:.2f}, MAE={mae:.2f}")

    for i, ax in enumerate(axes.flatten()):
        panel_label = chr(ord('A') + i)  # Convert index to corresponding letter
        ax.text(-0.06, 1.06, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')

    plt.tight_layout()
    plt.savefig('combination_tools_performance.pdf')
    plt.show()

    df_performance = pd.DataFrame({'method': method, 'PCC': pcc_list, 'SCC': scc_list, 'MAE':mae_list})
    
    return df_performance

def plot_bars_of_performance(df, parameter):
    
    if parameter == "MAE":
        df = df.sort_values(by=parameter)
    else:
        df = df.sort_values(by=parameter, ascending=False)
    
    methods = df['method']
    pcc_values = df[parameter]

    # Create bar plot
    plt.figure(figsize=(15, 6))
    plt.bar(methods, pcc_values)

    # Rotate x-axis labels
    plt.xticks(rotation=90)

    # Add labels and title
    plt.xlabel('Method')
    plt.ylabel(parameter)
    plt.title(f'{parameter} Values for Different Methods')

    # Show plot
    plt.tight_layout()
    plt.savefig(f'performance_bar_{parameter}.pdf')
    plt.show()


def get_correlations(df):
    column_list = ["ddg", "ddgun_seq", "ddgun_3d", "mutatex", "rasp", "rosetta", "thermompnn"]
    with open("correlations.csv", "w") as f:
        f.write("variable1, variable2, PCC, SCC, MAE\n")
        for i, column in enumerate(column_list):
            print(column)
            for other_column in column_list[i+1:]:  # Start from the index of the current column plus one
                print(other_column)
                df[column] = pd.to_numeric(df[column], errors='coerce')
                df[other_column] = pd.to_numeric(df[other_column], errors='coerce')
                df_temp = df[[column, other_column]].dropna()
                pearson_corr, _ = stats.pearsonr(df_temp[column], df_temp[other_column])
                spearman_corr, _ = stats.spearmanr(df_temp[column], df_temp[other_column])
                mae = mean_absolute_error(df_temp[column], df_temp[other_column])
                f.write(f"{column}, {other_column}, PCC:{pearson_corr}, SCC: {spearman_corr}, MAE: {mae}\n")

file = "../collect_data/collected_S615_data.csv"
df = pd.read_csv(file, index_col=0)

df[['AA_WT', 'pos', 'AA_MUT']] = df['AF_mut'].str.extract(r'([A-Z])(\d+)([A-Z])')
df['pos'] = pd.to_numeric(df['pos'])


#d = ['P02417', 'P00149', 'P0A3C7', 'P37957', 'Q5SIY4', 'P11961', 'P0A780', 'P00441', 'P02654', 'P18429', 'Q9GZQ8', 'P05112', 'Q53291', 'P51161']

#for i in range(len(d)):
    # Set the 'thermompnn' column to NaN where 'uniprot' matches the value in d
#    df.loc[df['uniprot'] == d[i], 'thermompnn'] = np.nan
#each parameter

print("getting correlation to each other")
get_correlations(df)
print("getting experimental correlation")
make_data_dist_figure(df)
print("get the performance in a plot")
df_performance = make_prediction_figure(df)

#Multiple linear combinations
#print("do multiple linear correlation part 1")
#df1, all_combinations1, df_coef1 = multiple_linear_correlations_part1(df, "./")
#print(df1.columns)
#print("do multiple linear correlation part 2")
#number = int(list(df_coef1.model)[-1].split(" ")[-1])
#df2, all_combinations2, df_coef2 = multiple_linear_correlations_part2(df, "./", number)
#print(df2.columns)
#
#df1.drop(columns=["thermompnn"], inplace=True)

# df = df1.merge(df2, on=['uniprot', 'AF_mut', 'AA_WT', 'pos', 'AA_MUT', 'ddg', 'effect',
#        'measure', 'method', 'ph', 'temperature', 'SS', 'pLDDT', 'gene',
#        'organism', 'human', 'DOI', 'mutatex', 'rosetta', 'rasp', 'ddgun_seq',
#        'ddgun_seq_cat', 'ddgun_3d', 'ddgun_3d_cat'], how='left')

# all_combinations = all_combinations1+all_combinations2

# df_coef = pd.concat([df_coef1, df_coef2])

# print("plot multiple linear correlation")
# df_performance_combi = plot_scatter_grid(df, all_combinations)

# df_all_performance = pd.concat([df_performance, df_performance_combi])

# df_all_performance.to_csv("all_model_performance.csv")
# df_new = df_performance_combi.merge(df_coef, on=["method"])

# df_new['PCC'] = df_new['PCC'].round(2)
# df_new['SCC'] = df_new['SCC'].round(2)
# df_new['MAE'] = df_new['MAE'].round(2)

# df_new.to_csv("table_S3.csv")

# df_all_performance = df_all_performance.sort_values(by="PCC", ascending=False)
# unwanted_methods = ["thermoMPNN", "MutateX", "Rosetta", "RaSP", "ddGun seq", "ddGun 3d"]
# # Filtering out rows with unwanted methods
# df_all_performance = df_all_performance[~df_all_performance["method"].isin(unwanted_methods)]
# sns.set(style="darkgrid", palette="viridis")
# plt.figure(figsize=(5, 15))  # Set the figure size
# sns.scatterplot(x=df_all_performance['PCC'], y=df_all_performance['method'])  # Adjust the size as needed
# plt.savefig("PCC_plot.pdf",bbox_inches='tight')
# plt.show()

#plot_bars_of_performance(df_all_performance, "PCC")
#plot_bars_of_performance(df_all_performance, "SCC")
#plot_bars_of_performance(df_all_performance, "MAE")
