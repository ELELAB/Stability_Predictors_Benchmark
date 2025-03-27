### casestudy 3 - CPA 2
path = "./"

import os
import pandas as pd
import re
import seaborn as sns
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from Bio import PDB
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

sns.set(style="dark", palette="viridis")

directories = ["heatmaps", "distribution_plots", "pdbs"]

for dir_name in directories:
    os.makedirs(dir_name, exist_ok=True)

######## MAVE ##########

#convert 3 letter amino acid to 1
aa_map = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"
}

removed_counts = {";": 0, "=": 0, "del/ins": 0}

def convert_hgvs(hgvs):
    
    # Check for removal conditions
    if "=" in hgvs:
        removed_counts["="] += 1
        return None
    if ";" in hgvs:
        removed_counts[";"] += 1
        return None
    if "del" in hgvs or "ins" in hgvs:
        removed_counts["del/ins"] += 1
        return None
    
    parts = hgvs[2:]  # Remove 'p.'
    
    # Extract the amino acid names and number
    match = re.match(r"([A-Za-z]+)(\d+)([A-Za-z]+)", parts)
    if not match:
        return None  # Skip if format is incorrect
    
    original_aa, position, new_aa = match.groups()
    
    # Convert amino acid names to single-letter codes
    if original_aa in aa_map and new_aa in aa_map:
        original_aa = aa_map[original_aa]
        new_aa = aa_map[new_aa]
    else:
        return None  # Skip if an amino acid is not found in the map

    # Adjust the position by +22
    new_position = int(position) + 22
    
    return f"{original_aa}{new_position}{new_aa}"

def prep_and_plot(mave_file, path, input_placement, ddg_file):
    mave = pd.read_csv(path+input_placement+mave_file)
    mave = mave[['hgvs_pro','score']]
    #normalization need
    WT_score = mave[:6].score.mean()
    #remove no mutations to avoid bias
    mave = mave[6:]
    
    mave = mave.dropna().reset_index(drop=True)
    #mave['RFS'] = mave['score']/abs(WT_score)
    mave['ddG']=(mave['score']-abs(WT_score))*-1
    
    currated_mutations = pd.read_csv(path+input_placement+"CPA2_mutations.csv")
    currated_mutations = currated_mutations[['AF_mut', 'ddg']]
    currated_mutations = currated_mutations.rename(columns={'AF_mut': 'Mutation'})
    #all mutations are available in the mave scores
    
    l_curration_subset = mave 
    l_curration_subset["Mutation"] = l_curration_subset["hgvs_pro"].apply(convert_hgvs)
    
    # Count rows before and after filtering
    l_curration_subset = mave.dropna(subset=['Mutation'])
    l = l_curration_subset.merge(currated_mutations, on="Mutation")

    # Perform linear regression using SciPy
    slope, intercept, r_value, p_value, std_err = linregress(l.ddG, l.ddg)  # Swap order
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Scatter plot
    ax1 = axes[0]
    sns.scatterplot(
        data=l, 
        x="ddG",  # Swapped to match linregress
        y="ddg", 
        label="Data Points",
        ax=ax1
    )
    
    # Generate line values
    x_range = np.linspace(-0.5, 3, 10)  # X range from -0.5 to 3
    y_range = slope * x_range + intercept  # Compute Y values
    corr, _ = pearsonr(l["ddG"], l["ddg"])  # Swapped order to match regression
    
    # Plot regression line
    ax1.plot(x_range, y_range, color='black', linestyle='--', 
             label=f"y = {slope:.2f}x + {intercept:.2f}, PCC={corr:.2f}")
    
    # Labels and title
    ax1.set_xlabel("ΔΔG MAVE")
    ax1.set_ylabel("ΔΔG S612")
    ax1.legend()
    ax1.set_title("Correlation of ddG from MAVE and S612", fontsize=14)
    
    # Plot 1: All alterations
    ax2 = axes[1]
    sns.histplot(mave.ddG, kde=True, ax=ax2)  # Fixed ax reference
    ax2.set_xlabel("ΔΔG MAVE")
    ax2.axvline(x=0.0, color='black', linestyle='dotted', linewidth=2)
    ax2.set_title(f"All {len(mave)} alterations", fontsize=14)
    
    # Apply mutation filtering
    mave["Mutation"] = mave["hgvs_pro"].apply(convert_hgvs)
    
    # Drop invalid mutations
    mave = mave.dropna(subset=['Mutation'])
    
    # Plot 2: Excluding deletions & double mutations
    ax3 = axes[2]
    sns.histplot(mave.ddG, kde=True, ax=ax3)  # Fixed ax reference
    ax3.axvline(x=0.0, color='black', linestyle='dotted', linewidth=2)
    ax3.set_xlabel("ΔΔG MAVE")
    ax3.set_title(f"Excluding indels ({removed_counts['del/ins']}) & double mutations ({removed_counts[';']})", fontsize=14)

    plt.tight_layout()
    plt.savefig(path+'distribution_plots/mave_distribution.pdf')
    plt.show()
    
    WT = []
    pos = []
    MUT = [] 
    for i in mave.Mutation:
        WT.append(i[0])
        pos.append(i[1:-1])
        MUT.append(i[-1])
        
    mave['WT']=WT
    mave['pos']=pos
    mave['MUT']=MUT

    heatmap_df = mave.pivot(index="MUT", columns="pos", values="ddG")

    plt.figure(figsize=(25, 5))
    ax = plt.axes()

    # Create heatmap with reversed colormap
    sns.heatmap(
        heatmap_df, 
        ax=ax,
        #cmap=sns.color_palette("vlag_r", as_cmap=True)  
        cmap="viridis"
    )

    title = "ΔΔG MAVE"
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path+'heatmaps/'+'ddG_mave_heatmap.pdf')
    plt.show()
        
    return mave, currated_mutations

# Apply the updated function to your DataFrame
mave, currated_mutations = prep_and_plot("urn_mavedb_00000164-0-1_scores.csv", path, "input_files/", "CPA2_mutations.csv")
# Apply the updated function to your DataFrame

##### Stability
def prep_mutatex(path, file):
    df=pd.read_csv(path+file)
    mutation_columns = df.columns[3:-3]
    data = []
    for index, row in df.iterrows():
        wt_residue = row['WT residue type']
        residue_number = row['Residue #']
        for mutation in mutation_columns:
            mutation_name = f"{wt_residue}{residue_number}{mutation}"
            mutatex_ddG = row[mutation]
            data.append([mutation_name, mutatex_ddG])
    # Create new DataFrame
    df_new = pd.DataFrame(data, columns=['Mutation', 'mutatex_ddG'])
    return df_new

def prep_rasp(path, file):
    df =pd.read_csv(path+file, index_col=0)
    if 'RaSP_ddG_avg' in df.columns:
        df = df[['variant', 'RaSP_ddG_avg']]
        df = df.rename(columns={'RaSP_ddG_avg': 'RaSP_ddG'})
    else:        
        df=df[['variant', 'RaSP_ddG']]
    df = df.rename(columns={'variant': 'Mutation'})
    return df

def prep_thermompnn(path,file):
    df =pd.read_csv(path+file, index_col=0)
    if 'ddG_pred_avg' in df.columns:
        df = df.reset_index()
        df['Mutation']=df['wildtype']+df['position'].astype(str)+df['mutation']
        df = df[['Mutation', 'ddG_pred_avg']]
        df = df.rename(columns={'ddG_pred_avg': 'Thermompnn_ddG'})
    else:
        df = df[['Mutation', 'ddG (kcal/mol)']]
        df = df.rename(columns={'ddG (kcal/mol)': 'Thermompnn_ddG'})
    return df
        
df_mutatex_simple = prep_mutatex(path+"input_files/", "CPA2_mutatex.csv")
df_mutatex_md = prep_mutatex(path+"input_files/", "CPA2_mutatex_md.csv")
df_mutatex_cabsflex = prep_mutatex(path+"input_files/", "CPA2_mutatex_cabsflex.csv")
df_rasp_simple = prep_rasp(path+"input_files/", "CPA2_rasp.csv")   
df_rasp_cabsflex = prep_rasp(path+"input_files/", "CPA2_rasp_cabsflex_20frames.csv")   
df_rasp_md = prep_rasp(path+"input_files/", "CPA2_RaSP_md_25.csv")   
df_thermo_simple=prep_thermompnn(path+"input_files/","CPA2_thermompnn.csv")
df_thermo_cabsflex=prep_thermompnn(path+"input_files/","CPA2_thermonpnn_cabsflex.csv")
df_thermo_md=prep_thermompnn(path+"input_files/","CPA2_thermonpnn_md.csv")

# Dictionary of dataframes with their corresponding names
dfs = {
    "df_mutatex_simple": df_mutatex_simple,
    "df_mutatex_cabsflex": df_mutatex_cabsflex,
    "df_mutatex_md": df_mutatex_md,
    "df_rasp_simple": df_rasp_simple,
    "df_rasp_cabsflex": df_rasp_cabsflex,
    "df_rasp_md": df_rasp_md,
    "df_thermo_simple": df_thermo_simple,
    "df_thermo_cabsflex": df_thermo_cabsflex,
    "df_thermo_md": df_thermo_md
}

# Rename the second column dynamically
for name, df in dfs.items():
    col_name = df.columns[1]  # Get the second column name
    new_col_name = f"{col_name}_{'_'.join(name.split('_')[2:])}"  # Append the dataframe suffix
    df.rename(columns={col_name: new_col_name}, inplace=True)


# Merge all dataframes on "Mutation"
merged_df = list(dfs.values())[0]  # Start with the first dataframe
for df in list(dfs.values())[1:]:
    merged_df = merged_df.merge(df, on="Mutation", how="outer")
    
merged_df = merged_df.merge(mave[['Mutation', 'ddG']], on="Mutation", how="outer")
merged_df = merged_df.merge(currated_mutations, on="Mutation", how="outer")

WT = []
pos = []
MUT = [] 
for i in merged_df.Mutation:
    WT.append(i[0])
    pos.append(i[1:-1])
    MUT.append(i[-1])
    
merged_df['WT']=WT
merged_df['pos']=pos
merged_df['MUT']=MUT

merged_df['pos']=merged_df['pos'].astype(int)
merged_df = merged_df[merged_df['pos']>=23]
fitness_range = merged_df[merged_df['pos']<=94]

#Heatmaps in the fitness range
def make_heatmap(df, column):
    heatmap_df = df.pivot(index="MUT", columns="pos", values=column)
    
    plt.figure(figsize=(25, 5))
    ax = plt.axes()
    
    # Create heatmap with reversed colormap
    sns.heatmap(
        heatmap_df, 
        ax=ax,
        cmap="viridis"
        #cmap=sns.color_palette("vlag", as_cmap=True)  
    )
    
    title = (" ").join(column.split("_"))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path+'heatmaps/'+f"heatmap_{column}.pdf")
    
    plt.show()
    
for i in fitness_range.columns[1:-5]:
    make_heatmap(fitness_range, i)
    
def make_normalized_heatmap(df, column, scaler_name):
    heatmap_df = df.pivot(index="MUT", columns="pos", values=column)
    
    #Best for: Preserving relative differences while standardizing the scale.
    #This rescales all values between 0 and 1 based on the min and max values.
    if scaler_name == "min_max":
        scaler = MinMaxScaler()
        heatmap_df = pd.DataFrame(scaler.fit_transform(heatmap_df), columns=heatmap_df.columns, index=heatmap_df.index)
    
    #Best for: Comparing across datasets with different distributions.
    #This converts values into a standard normal distribution (mean = 0, std = 1).
    if scaler_name == "z-score_normalization":
        scaler = StandardScaler()
        heatmap_df = pd.DataFrame(scaler.fit_transform(heatmap_df), columns=heatmap_df.columns, index=heatmap_df.index)
    
    #Best for: Data with outliers that could distort min-max or z-score scaling.
    #This scales based on the median and IQR, making it robust to outliers.
    if scaler_name == "Robust_scaling":
        scaler = RobustScaler()
        heatmap_df = pd.DataFrame(scaler.fit_transform(heatmap_df), columns=heatmap_df.columns, index=heatmap_df.index)
    
    plt.figure(figsize=(25, 5))
    ax = plt.axes()
    
    # Create heatmap with reversed colormap
    sns.heatmap(
        heatmap_df, 
        ax=ax,
        cmap="viridis"
        #cmap=sns.color_palette("vlag", as_cmap=True)  
    )
    
    title = f"{(' ').join(column.split('_'))} normalized by {scaler_name}"
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path+'heatmaps/'+f"heatmap_normalized_{column}_{scaler_name}.pdf")
    
    plt.show()
    
#for i in fitness_range.columns[1:-5]:
#    make_normalized_heatmap(fitness_range, i, "min_max")
#    make_normalized_heatmap(fitness_range, i, "z-score_normalization")
#    make_normalized_heatmap(fitness_range, i, "Robust_scaling")

ddg_df = merged_df.dropna() 
ddg_df.to_csv("S612_CPA2_predictions.csv")
#leaves the 20 mutations where there are measurements

# Define columns of interest
predicted_cols = [
    "mutatex_ddG_simple",
    "mutatex_ddG_cabsflex",
    "mutatex_ddG_md",
    "RaSP_ddG_simple",
    "RaSP_ddG_cabsflex",
    "RaSP_ddG_md",
    "Thermompnn_ddG_simple",
    "Thermompnn_ddG_cabsflex",
    "Thermompnn_ddG_md",
]

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

# Loop through each column and create scatter plot
for i, col in enumerate(predicted_cols):
    ax = axes[i]
    sns.scatterplot(x=ddg_df['ddg'], y=ddg_df[col], ax=ax)
    
    # Compute Pearson correlation coefficient
    corr, _ = pearsonr(ddg_df['ddg'], ddg_df[col])
    print(f"{col} has a PCC of {corr} toward S612 ddG")
    
    # Set labels and title
    ax.set_xlabel("Experimental ΔΔG (kcal/mol)")
    ax.set_ylabel("Predicted ΔΔG (kcal/mol)")
    ax.set_title(" ".join(col.split("_")))
    
    # Annotate with correlation coefficient
    ax.text(0.05, 0.9, f'Pearson r: {corr:.2f}', transform=ax.transAxes, fontsize=12)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig(path+'distribution_plots/'+'correlations_ddg.pdf')
plt.show()

fitness_range = fitness_range.dropna(subset="ddG")
fitness_range['mutatex_thermompnn']=(fitness_range['mutatex_ddG_md'] + fitness_range['Thermompnn_ddG_md']) /2
fitness_range['mutatex_rasp']=(fitness_range['mutatex_ddG_md'] + fitness_range['RaSP_ddG_md']) /2
fitness_range['thermompnn_rasp']=(fitness_range['RaSP_ddG_md'] + fitness_range['Thermompnn_ddG_md']) /2
fitness_range['mutatex_rasp_thermompnn']=(fitness_range['mutatex_ddG_md'] + fitness_range['RaSP_ddG_md'] + fitness_range['Thermompnn_ddG_md']) /3

corr, _ = pearsonr(fitness_range['ddG'], fitness_range["mutatex_thermompnn"])
print(f"mutatex_thermompnn has a PCC of {corr} toward MAVE ΔΔG")
corr, _ = pearsonr(fitness_range['ddG'], fitness_range["mutatex_rasp"])
print(f"mutatex_rasp has a PCC of {corr} toward MAVE ΔΔG")
corr, _ = pearsonr(fitness_range['ddG'], fitness_range["thermompnn_rasp"])
print(f"thermompnn_rasp has a PCC of {corr} toward MAVE ΔΔG")
corr, _ = pearsonr(fitness_range['ddG'], fitness_range["mutatex_rasp_thermompnn"])
print(f"mutatex_rasp_thermompnn has a PCC of {corr} toward MAVE ΔΔG")

# Create subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

# Loop through each column and create scatter plot
for i, col in enumerate(predicted_cols):
    ax = axes[i]
    sns.scatterplot(x=fitness_range['ddG'], y=fitness_range[col], ax=ax, alpha=0.6)
    
    # Compute Pearson correlation coefficient
    corr, _ = pearsonr(fitness_range['ddG'], fitness_range[col])
    print(f"{col} has a PCC of {corr} toward MAVE ΔΔG")
    
    # Set labels and title
    ax.set_xlabel("ΔΔG (kcal/mol) MAVE")
    ax.set_ylabel("Predicted ΔΔG (kcal/mol)")
    subtitle = " ".join(col.split("_"))
    title = f"{subtitle[0]} ΔΔG (kcal/mol) {subtitle[-1]}"
    ax.set_title(title)
    
    # Annotate with correlation coefficient
    ax.text(0.05, 0.9, f'Pearson r: {corr:.2f}', transform=ax.transAxes, fontsize=12)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig(path+'distribution_plots/'+'correlations_ddg_mave.pdf')
plt.show()

############### trying to make the scores the b-factor
#use fitness_range
#make a smaller dataframe

position_list = []
calculator = []
avg_value = []

for position in set(fitness_range.pos):
    for column in fitness_range.columns[1:-9]:
        position_list.append(position)
        calculator.append(column)
        avg_value.append(fitness_range[fitness_range.pos == position][column].mean())

b_factor_df = pd.DataFrame({'position':position_list, 'method':calculator, 'site_value':avg_value})
        
# Define input and output paths
input_pdb_path = path + "input_files/" + "CPA2_littledomain.pdb"
output_pdb_template = path + "pdbs/" + "CPA2_{}.pdb"

# Load PDB file
parser = PDB.PDBParser(QUIET=True)
structure = parser.get_structure("CPA2", input_pdb_path)

# Iterate over each method
for method in b_factor_df["method"].unique():
    # Filter data for the current method
    method_df = b_factor_df[b_factor_df["method"] == method]
    position_to_bfactor = dict(zip(method_df["position"], method_df["site_value"]))
    
    # Update B-factor values
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()[1]  # Get residue position
                if res_id in position_to_bfactor:
                    for atom in residue:
                        atom.set_bfactor(position_to_bfactor[res_id])
    
    # Save updated PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_template.format(method))

fitness_range = fitness_range[['Mutation','mutatex_ddG_md', 
                               'RaSP_ddG_md', 'Thermompnn_ddG_md',
                               'ddG', 'WT', 'pos', 'MUT']]

# Define columns to average
ddG_cols = ['mutatex_ddG_md', 'RaSP_ddG_md', 'Thermompnn_ddG_md', 'ddG']

# Compute the mean ddG values for each mutation type
avg_fitness = fitness_range.groupby("MUT")[ddG_cols].mean().reset_index()

# Reshape the data for easier plotting
avg_fitness_melted = avg_fitness.melt(id_vars="MUT", var_name="Model", value_name="Average ΔΔG")

# Plot using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(data=avg_fitness_melted, x="MUT", y="Average ΔΔG", hue="Model", palette="viridis")

# Labels and title
plt.xlabel("Mutant Residue (MUT)")
plt.ylabel("Average ΔΔG")
plt.title("Average ΔΔG values by Mutant Residue")
plt.legend(title="Model")

# Show the plot
plt.xticks()  # Rotate x-axis labels if needed
plt.tight_layout()
plt.savefig(path+'distribution_plots/'+'per_residue_difference.pdf')
plt.show()


