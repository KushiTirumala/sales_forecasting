import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load CSV sales data."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_column):
    """Preprocess data: handle missing values, encode categorical features, scale numeric."""
    
    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
