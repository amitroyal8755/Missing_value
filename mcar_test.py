import pandas as pd
from scipy.stats import chi2_contingency, combine_pvalues
import numpy as np
def test_mcar(df, col):
    df1 = df.copy()
    df1["missing_indicator"] = df1[col].isnull().astype(int)
    columns_to_test = df1.columns.drop([col, "missing_indicator"])

    p_values = []

    for i in columns_to_test:
        # print(f"Processing column: {i}, Type: {df1[i].dtype}")
        if df1[i].isnull().any():
            if df1[i].dtype.name == 'category':
                df1[i] = df1[i].cat.add_categories('missing').fillna('missing')
            else:
                df1[i] = df1[i].fillna("missing")

        contingency_table = pd.crosstab(df1[i], df1["missing_indicator"])
        _, p, _, _ = chi2_contingency(contingency_table)
        p_values.append(p)

    combined_p_value = combine_pvalues(p_values)[1]

    return combined_p_value
