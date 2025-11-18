import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from hyperopt import hp
from hyperopt.pyll.base import scope

# --- CLASE 1: DATA WRANGLER ---
class DataWrangler(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado para:
    1. Llenar NaNs en 'Health_Issues' con 'None' y codificarla.
    2. Mapear 'Country' a columnas de continentes.
    3. Eliminar 'ID' y convertir tipos de 'Smoking'/'Alcohol'.
    """
    def __init__(self):
        self.countries_to_continents = {
            "Europe": ["Germany", "Spain", "France", "UK", "Switzerland", "Netherlands",
                       "Italy", "Belgium", "Finland", "Sweden", "Norway"], # Corregí "Sweeden"
            "Asia": ["China", "Japan", "South Korea", "India"],
            "America": ["Brazil", "Mexico", "Canada", "USA"],
            "Oceania": ["Australia"]
        }
        self.continent_cols_ = ["is_europe", "is_asia", "is_america", "is_oceania"]
        self.ordinal_categories_list = [["Severe", "Moderate", "Mild", "None"]]
        self.ordinal_encoder = OrdinalEncoder(categories=self.ordinal_categories_list)

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        X_filled = X[["Health_Issues"]].fillna("None")
        self.ordinal_encoder.fit(X_filled)
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_ = X.copy()
        
        if 'Country' in X_.columns:
            X_["is_europe"] = X_['Country'].apply(lambda x: 1 if x in self.countries_to_continents["Europe"] else 0)
            X_["is_asia"] = X_['Country'].apply(lambda x: 1 if x in self.countries_to_continents["Asia"] else 0)
            X_["is_america"] = X_['Country'].apply(lambda x: 1 if x in self.countries_to_continents["America"] else 0)
            X_["is_oceania"] = X_['Country'].apply(lambda x: 1 if x in self.countries_to_continents["Oceania"] else 0)
            X_ = X_.drop("Country", axis=1)
            
        if "Smoking" in X_.columns:
            X_["Smoking"] = X_["Smoking"].astype(bool)
        if "Alcohol_Consumption" in X_.columns:
            X_["Alcohol_Consumption"] = X_["Alcohol_Consumption"].astype(bool)
            
        if "ID" in X_.columns:
            X_ = X_.drop("ID", axis=1)
            
        if "Health_Issues" in X_.columns:
            X_filled_transformed = X_[["Health_Issues"]].fillna("None")
            X_["Health_Issues"] = self.ordinal_encoder.transform(X_filled_transformed)
            
        return X_
    
    def get_feature_names_out(self, input_features=None) -> np.array:
        if input_features is None:
            input_features = self.feature_names_in_
        output_features = list(input_features)
        
        if "Country" in output_features:
            output_features.remove("Country")
            output_features.extend(self.continent_cols_)
        if "ID" in output_features:
            output_features.remove("ID")
                        
        return np.array(output_features, dtype=object)

# --- CLASE 2: COLUMN DROPPER ---
class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Un transformador personalizado para eliminar columnas específicas.
    """
    def __init__(self, columns_to_drop: list[str]) -> None:
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        # Guarda las features de entrada para usarlas en get_feature_names_out
        self.feature_names_in_ = list(X.columns)
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnDropper solo funciona con DataFrames de pandas.")
        return X.drop(columns=self.columns_to_drop, errors='ignore')

    def get_feature_names_out(self, input_features=None) -> np.array:
        """
        Genera los nombres de las columnas de salida después de eliminar.
        """
        if input_features is None:
            input_features = self.feature_names_in_
            
        output_features = [
            col for col in input_features 
            if col not in self.columns_to_drop
        ]
        return np.array(output_features, dtype=object)

# --- CONSTANTES: LISTAS DE COLUMNAS ---
NUMERIC_COLS = ['Age','Coffee_Intake', 'Caffeine_mg', 'Sleep_Hours', 'BMI', 'Heart_Rate', 'Physical_Activity_Hours']
WRANGLING_COLS = [ "Country", "Smoking", "Alcohol_Consumption", "ID", "Health_Issues"]
ORDINAL_CATEGORIES_COLS = ["Stress_Level"]
ONE_HOT_CATEGORIES_COLS = ["Gender", "Occupation"]
CUSTOM_ORDINAL_CATEGORIES = [["High", "Medium", "Low"]] # Para Stress_Level

# --- FUNCIÓN "CONSTRUCTORA" DEL PIPELINE ---
def create_preprocessing_pipeline(columns_to_drop: list[str]) -> Pipeline:
    """
    Construye y devuelve el pipeline de preprocesamiento completo.
    """
    ct = ColumnTransformer([
        ("Data_wrangling", DataWrangler(), WRANGLING_COLS),
        ("robust_scaler", RobustScaler(), NUMERIC_COLS),
        ("ordinal_encoding", OrdinalEncoder(categories=CUSTOM_ORDINAL_CATEGORIES), ORDINAL_CATEGORIES_COLS),
        ("onehot_encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONE_HOT_CATEGORIES_COLS),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False)
    
    ct.set_output(transform="pandas")

    full_preprocessing = Pipeline([
        ("preprocessing", ct),
        ("dropper", ColumnDropper(columns_to_drop=columns_to_drop))
    ])
    
    return full_preprocessing