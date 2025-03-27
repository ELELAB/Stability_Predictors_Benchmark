#cleanup and combine

import pandas as pd
import itertools

path = "../collect_data/"

df = pd.read_csv(path+"collected_S615_data.csv", index_col=0)
df = df.dropna(subset=["ddgun_3d"])

columns = ['mutatex', 'rosetta', 'rasp', 'ddgun_seq', 'ddgun_3d', 'thermompnn']

# Function to create new column based on combination of columns
def create_combinations(df, columns, comb_size):
    for combo in itertools.combinations(columns, comb_size):
        col_name = '_'.join(combo)
        df[col_name] = df[list(combo)].astype(float).mean(axis=1)
    return df

df = create_combinations(df, columns, 2)
df = create_combinations(df, columns, 3)
df = create_combinations(df, columns, 4)

# Calculate specific averages as needed
df['average_all'] = df[columns].astype(float).mean(axis=1)
df['average_nonthermo'] = df[columns[:-1]].astype(float).mean(axis=1)

###############################
# step 1: CLEAN DATA
###############################
df[['AA', 'pos', 'MT']] = df['AF_mut'].str.extract(r'([A-Za-z]+)(\d+)([A-Za-z]+)')

for i in df.columns:
    if type(df[i].iloc[0])==str:
        if df[i].iloc[0].endswith(('a', 'b', 'c')):
            new_column_name = f"{i}_sequencesim"
            df[new_column_name] = df[i].str[-1]
            df[i] = df[i].str.slice(stop=-1)
            df[i] = pd.to_numeric(df[i], errors='coerce')

df.to_csv("combined_S612.csv")
