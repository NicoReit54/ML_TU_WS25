import pandas as pd

# One of the enery datafiles

df = pd.read_csv("Exercise2/data/WEC_Perth_49.csv")
df.head()
df = df.dropna()

target_col = 'Total_Power'
if target_col in df.columns:
    feature_df = df.drop(columns=[target_col])
    y = df[target_col].values
 
num_cols = feature_df.select_dtypes(include=['number']).columns

X = feature_df[num_cols].values

result_df = pd.DataFrame(X, columns=num_cols)
result_df[target_col] = y


result_df.to_csv("Exercise2/data/wave_energy_farm_preprocessed.csv", index=False)
