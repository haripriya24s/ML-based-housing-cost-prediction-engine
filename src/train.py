# train.py

import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import load_data, preprocess_data


def train_model(data_path):
    # Load dataset
    df = load_data(data_path)

    # Preprocess data
    X, y = preprocess_data(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    joblib.dump(model, "../models/house_price_model.pkl")

    print("Model saved successfully!")

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Performance:")
    print("MSE:", mse)
    print("R2 Score:", r2)

    return model


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved at {path}")


if __name__ == "__main__":
    # Path to dataset
    data_path = "../data/train.csv"

    # Train model
    model = train_model(data_path)

    # Save model
    save_model(model, "../models/house_price_model.pkl")