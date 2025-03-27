import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error
#sns.set_palette("viridis")

sns.set(style="dark", palette="viridis")

df = pd.read_csv("combined_S612.csv", index_col=0)

with open('checking_sequence_thermompnn/too_similar.txt') as f:
    uniprot_list = [line.rstrip('\n') for line in f]

in_training = []

for uniprot in df.uniprot: 
    print(str(uniprot))
    if str(uniprot) in uniprot_list:
        in_training.append("True")
    else:
        in_training.append("False")
        
df['ThermoMPNN>=25%sim'] = in_training
 

def make_prediction_figure(df):
    
    # Filtering limited data
    df['ThermoMPNN>=25%sim']= df['ThermoMPNN>=25%sim'].astype(str)
    
    fig, axes = plt.subplots(3, 2, figsize=(11, 15))
    
    # The rest of the methods, including limited data for each:
    methods_data = [
        ("ThermoMPNN", "thermompnn", axes[0,0]),
        ("ddGun seq", "ddgun_seq", axes[0, 1]),
        ("ddGun 3d", "ddgun_3d", axes[1, 0]),
        ("MutateX", "mutatex", axes[1, 1]),
        ("Rosetta", "rosetta", axes[2, 0]),
        ("RaSP", "rasp", axes[2, 1])
    ]
    
    for method_name, column, ax in methods_data:
        # Drop NAs for the current method
        df_temp = df[["ddg", column,"ThermoMPNN>=25%sim"]].dropna()
        
        # Scatter plot
        sns.scatterplot(data=df_temp, x='ddg', y=column, ax=ax, alpha=0.4, hue="ThermoMPNN>=25%sim")
        ax.set_xlabel('Experimental ΔΔG in kcal/mol')
        ax.set_ylabel('Predicted ΔΔG in kcal/mol')
        ax.plot([-2, 9], [-2, 9], '--', color='black', linewidth=1)
        ax.set_ylim(-4, 10)
                
        # Title
        ax.set_title(f"{method_name}")
        
    for i, ax in enumerate(axes.flatten()):
        panel_label = chr(ord('A') + i)
        ax.text(-0.06, 1.06, panel_label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.savefig('core_tools_performance.pdf')
    plt.show()
    
    # Final DataFrame with both limited and full data    
    return 

make_prediction_figure(df)

def calculate_performance_metrics(df):
    
    method = []
    pcc_list = []
    pcc_list_limited = []
    scc_list = []
    scc_list_limited = []
    mae_list = []
    mae_list_limited = []
    
    # Filtering limited data
    df['ThermoMPNN>=25%sim']= df['ThermoMPNN>=25%sim'].astype(str)
    
    columns_of_interest = ['mutatex',
           'rosetta', 'rasp', 'ddgun_seq', 'ddgun_3d',
           'thermompnn', 'mutatex_rosetta', 'mutatex_rasp',
           'mutatex_ddgun_seq', 'mutatex_ddgun_3d', 'mutatex_thermompnn',
           'rosetta_rasp', 'rosetta_ddgun_seq', 'rosetta_ddgun_3d',
           'rosetta_thermompnn', 'rasp_ddgun_seq', 'rasp_ddgun_3d',
           'rasp_thermompnn', 'ddgun_seq_ddgun_3d', 'ddgun_seq_thermompnn',
           'ddgun_3d_thermompnn', 'mutatex_rosetta_rasp',
           'mutatex_rosetta_ddgun_seq', 'mutatex_rosetta_ddgun_3d',
           'mutatex_rosetta_thermompnn', 'mutatex_rasp_ddgun_seq',
           'mutatex_rasp_ddgun_3d', 'mutatex_rasp_thermompnn',
           'mutatex_ddgun_seq_ddgun_3d', 'mutatex_ddgun_seq_thermompnn',
           'mutatex_ddgun_3d_thermompnn', 'rosetta_rasp_ddgun_seq',
           'rosetta_rasp_ddgun_3d', 'rosetta_rasp_thermompnn',
           'rosetta_ddgun_seq_ddgun_3d', 'rosetta_ddgun_seq_thermompnn',
           'rosetta_ddgun_3d_thermompnn', 'rasp_ddgun_seq_ddgun_3d',
           'rasp_ddgun_seq_thermompnn', 'rasp_ddgun_3d_thermompnn',
           'ddgun_seq_ddgun_3d_thermompnn', 'mutatex_rosetta_rasp_ddgun_seq',
           'mutatex_rosetta_rasp_ddgun_3d', 'mutatex_rosetta_rasp_thermompnn',
           'mutatex_rosetta_ddgun_seq_ddgun_3d',
           'mutatex_rosetta_ddgun_seq_thermompnn',
           'mutatex_rosetta_ddgun_3d_thermompnn',
           'mutatex_rasp_ddgun_seq_ddgun_3d', 'mutatex_rasp_ddgun_seq_thermompnn',
           'mutatex_rasp_ddgun_3d_thermompnn',
           'mutatex_ddgun_seq_ddgun_3d_thermompnn',
           'rosetta_rasp_ddgun_seq_ddgun_3d', 'rosetta_rasp_ddgun_seq_thermompnn',
           'rosetta_rasp_ddgun_3d_thermompnn',
           'rosetta_ddgun_seq_ddgun_3d_thermompnn',
           'rasp_ddgun_seq_ddgun_3d_thermompnn']
    
    df_lim = df[df['ThermoMPNN>=25%sim'] == "False"]
    
    for column in columns_of_interest:
        # Drop NAs for the current method
        df_temp = df[["ddg", column,"ThermoMPNN>=25%sim"]].dropna()
        df_temp_lim = df_lim[["ddg", column,"ThermoMPNN>=25%sim"]].dropna()


        pearson_corr, _ = stats.pearsonr(df_temp['ddg'], df_temp[column])
        spearman_corr, _ = stats.spearmanr(df_temp['ddg'], df_temp[column])
        mae = mean_absolute_error(df_temp['ddg'], df_temp[column])
        
        method.append(column)
        pcc_list.append(pearson_corr)
        scc_list.append(spearman_corr)
        mae_list.append(mae)
        
        # Limited data metrics
        pearson_corr, _ = stats.pearsonr(df_temp_lim['ddg'], df_temp_lim[column])
        spearman_corr, _ = stats.spearmanr(df_temp_lim['ddg'], df_temp_lim[column])
        mae = mean_absolute_error(df_temp_lim['ddg'], df_temp_lim[column])
        
        pcc_list_limited.append(pearson_corr)
        scc_list_limited.append(spearman_corr)
        mae_list_limited.append(mae)
        
    # Final DataFrame with both limited and full data
    df_performance = pd.DataFrame({
        'method': method,
        'PCC': pcc_list,
        'SCC': scc_list,
        'MAE': mae_list,
        'PCC_limited': pcc_list_limited,
        'SCC_limited': scc_list_limited,
        'MAE_limited': mae_list_limited})
    
    return df_performance

df_performance=calculate_performance_metrics(df)

df_performance.to_csv("performance_metrics_individual_tools.csv")

#flatten
data = pd.melt(df_performance, id_vars=['method'], value_vars=['PCC','SCC', 'MAE', 'PCC_limited', 'SCC_limited','MAE_limited'],
                        var_name='metric', value_name='value')

def plot_scatter(data):
    # Separate the 'full dataset' from the rest
    variables = ['PCC', 'SCC']
    for variable in variables:
        df = data[(data.metric == variable) | (data.metric == f"{variable}_limited")]
        
        df = df.sort_values(by="value")

        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x="method", y="value", alpha=0.7, hue='metric')
        plt.xticks(rotation=90)
        plt.ylim(0.0, 0.75)
        plt.ylabel(f"{variable} value")
        plt.tight_layout()
        plt.savefig(f"scatter_plot_{variable}.pdf")
        plt.show()

plot_scatter(data)
