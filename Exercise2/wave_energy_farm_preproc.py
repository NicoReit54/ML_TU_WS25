import pandas as pd

# Mostly taken from Lenard's exploration notebook. Did not check much if it makes sense or is proper, so should be double checked and fixed.

df = pd.read_csv("data/Steel_industry_data.csv")
df.head()
df = df.drop(['date'],axis=1)
df = df.dropna()

target_col = 'Total_Power'
if target_col in df.columns:
    feature_df = df.drop(columns=[target_col])
    y = df[target_col].values
 
num_cols = feature_df.select_dtypes(include=['number']).columns

X = feature_df[num_cols].values

result_df = pd.DataFrame(X, columns=num_cols)
result_df[target_col] = y


result_df.to_csv("data/steel_industry_data_preprocessed.csv", index=False)
