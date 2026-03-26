def load_data(file_path):
    """Load data from a CSV file."""
    import pandas as pd
    return pd.read_csv(file_path)


def preprocess_data(data):
    """Perform data preprocessing steps on the raw data."""
    # Example preprocessing steps: handle missing values, normalization, etc.
    data.fillna(method='ffill', inplace=True)
    return data


def load_model(model_path):
    """Load a machine learning model from a file."""
    import joblib
    return joblib.load(model_path)


def make_prediction(model, data):
    """Make predictions using the loaded model and the preprocessed data."""
    return model.predict(data)