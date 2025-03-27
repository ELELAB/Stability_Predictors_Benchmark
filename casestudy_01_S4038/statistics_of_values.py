import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind

# Load your data
df = pd.read_csv("PCCs.csv", index_col=0)

# Separate data by 'random subset' and 'full dataset'
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
    
df_pval.to_csv("pvalues.csv")
