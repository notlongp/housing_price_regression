import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def get_clean_data(X):
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(exclude = ["object"]).columns
    
    categorical_indices = np.argwhere(X.columns.isin(X.select_dtypes(include = ["object"])))
    numerical_indices = np.argwhere(X.columns.isin(X.select_dtypes(exclude = ["object"])))
    
    label_encode_col = ['Street', "Utilities", "CentralAir"]
    onehot_encode_col = X.iloc[:, categorical_indices.reshape(-1)].columns[X.iloc[:, categorical_indices.reshape(-1)].columns.isin(label_encode_col) == False]
    onehot_encode_col = list(onehot_encode_col)
    
    
    # Label Encoding
    for col in label_encode_col:
        lb = LabelEncoder()
        X.loc[:, col] = lb.fit_transform(X.loc[:, col].astype(str))
        
    # One Hot Encoding
    X.loc[:, onehot_encode_col] = X.loc[:, onehot_encode_col].fillna("NoneGiven")

    onehot_df = pd.get_dummies(X.loc[:, onehot_encode_col],drop_first=True)
    
    new_X = pd.concat([X.loc[:, numerical_features], X.loc[:, label_encode_col], onehot_df], axis=1)
    
    # Converting Year into TimeFromYear
    years_col = ['YearBuilt', 'YearRemodAdd', 'YrSold']
    for col in years_col:
        new_X.loc[:, col] = 2020 - new_X.loc[:, col]
        
    new_X = new_X.apply(lambda x: x.fillna(x.mean()),axis=0)
        
    return new_X
