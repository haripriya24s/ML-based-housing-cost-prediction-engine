import pandas as pd

# Load dataset
def load_data(path):
    data = pd.read_csv(path)
    return data

# Preprocess dataset
def preprocess_data(data):
    # Select only needed features
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
    target = 'SalePrice'

    X = data[features]
    y = data[target]

    # Handle missing values
    X = X.fillna(X.mean())

    return X, y