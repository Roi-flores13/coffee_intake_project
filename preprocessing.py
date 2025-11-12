import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder

from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

numeric_cols = ['Age','Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours', 'BMI', 'Heart_Rate', 'Physical_Activity_Hours'] # Variables that have to be scaled

ordinal_categories = ["Stress_Level", "Sleep_Quality", "Health_Issues"] # Variables that have a hierarchical order

one_hot_categories = ["Gender", "Occupation"] # Variables that just need one-hot encoding

passthorugh = ["Smoking", "Alcohol_Consumption", "is_asia", "is_america", "is_europe"] # Variables that are already fine as they are

custom_categories = [["High", "Medium", "Low"], # Stress Level hierarchical order
                     ["Poor", "Fair", "Good", "Excellent"], # Sleep Quality hierarchical order
                     ["Severe", "Moderate", "Mild", "None"]] # Health Issues hierarchical order


ct = ColumnTransformer([
    ("robust_scaler", RobustScaler(), numeric_cols), # scaler
    ("ordinal_encoding", OrdinalEncoder(categories=custom_categories), ordinal_categories), #ordinal encoding
    ("onehot_encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False), one_hot_categories), #one hot encoding
    ("passthrough", "passthrough", passthorugh) #ignore
],
remainder="drop")