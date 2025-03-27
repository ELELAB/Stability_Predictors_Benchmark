import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error
import glob as glob
sns.set_palette("viridis")

path = "./"
df = pd.read_csv(path+"combined_S612_thermompnn25.csv", index_col=0)
df_full = df[df['ThermoMPNN>=25%sim'] == False]

def get_names(df):
    df = df[['ddg', 'mutatex','rosetta', 'rasp', 'ddgun_seq', 'ddgun_3d',
       'thermompnn','mutatex_rosetta','mutatex_rasp','mutatex_ddgun_seq',
       'mutatex_ddgun_3d','mutatex_thermompnn','rosetta_rasp','rosetta_ddgun_seq',
       'rosetta_ddgun_3d','rosetta_thermompnn','rasp_ddgun_seq','rasp_ddgun_3d',
       'rasp_thermompnn','ddgun_seq_ddgun_3d','ddgun_seq_thermompnn',
       'ddgun_3d_thermompnn','mutatex_rosetta_rasp','mutatex_rosetta_ddgun_seq',
       'mutatex_rosetta_ddgun_3d','mutatex_rosetta_thermompnn','mutatex_rasp_ddgun_seq',
       'mutatex_rasp_ddgun_3d','mutatex_rasp_thermompnn','mutatex_ddgun_seq_ddgun_3d',
       'mutatex_ddgun_seq_thermompnn','mutatex_ddgun_3d_thermompnn','rosetta_rasp_ddgun_seq',
       'rosetta_rasp_ddgun_3d','rosetta_rasp_thermompnn','rosetta_ddgun_seq_ddgun_3d',
       'rosetta_ddgun_seq_thermompnn','rosetta_ddgun_3d_thermompnn','rasp_ddgun_seq_ddgun_3d',
       'rasp_ddgun_seq_thermompnn','rasp_ddgun_3d_thermompnn','ddgun_seq_ddgun_3d_thermompnn',
       'mutatex_rosetta_rasp_ddgun_seq','mutatex_rosetta_rasp_ddgun_3d','mutatex_rosetta_rasp_thermompnn',
       'mutatex_rosetta_ddgun_seq_ddgun_3d','mutatex_rosetta_ddgun_seq_thermompnn','mutatex_rosetta_ddgun_3d_thermompnn',
       'mutatex_rasp_ddgun_seq_ddgun_3d','mutatex_rasp_ddgun_seq_thermompnn','mutatex_rasp_ddgun_3d_thermompnn',
       'mutatex_ddgun_seq_ddgun_3d_thermompnn','rosetta_rasp_ddgun_seq_ddgun_3d','rosetta_rasp_ddgun_seq_thermompnn',
       'rosetta_rasp_ddgun_3d_thermompnn','rosetta_ddgun_seq_ddgun_3d_thermompnn','rasp_ddgun_seq_ddgun_3d_thermompnn',
       'average_all']]    
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


full_data_long.to_csv("PCCs_332_long.csv")
