import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Mostly taken from Lenard's exploration notebook. Did not check much if it makes sense or is proper, so should be double checked and fixed.

df = pd.read_csv("Exercise2/data/corn_data.csv")
df.head()
df = df.drop(['County','Farmer','Crop','Power source','Main credit source','Crop insurance','Farm records','Main advisory source','Advisory format',
              'Advisory language','Extension provider','Latitude', 'Longitude'],axis=1)
df = df.dropna()

# separate features and target
target_col = 'Yield'
if target_col in df.columns:
    feature_df = df.drop(columns=[target_col])
    y = df[target_col].values

# prepare for onehot encoding
cat_cols = feature_df.select_dtypes(include=['object']).columns
num_cols = feature_df.select_dtypes(exclude=['object']).columns

# One-Hot Encoding
print(f"Encoding categorical columns: {list(cat_cols)}")
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cats = encoder.fit_transform(feature_df[cat_cols])

# Combine Numeric + Encoded Features
X_num = feature_df[num_cols].values
X = np.hstack([X_num, encoded_cats])

#put together final DataFrame for saving
result_df = pd.DataFrame(X, columns=list(num_cols) + list(encoder.get_feature_names_out(cat_cols)))
result_df[target_col] = y

#result_df.head()
#print(result_df.dtypes)

result_df.to_csv("Exercise2/data/corn_data_preprocessed.csv", index=False)
