import numpy as np
import joblib

def load_model():
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "house_price_model.pkl")

   
    return joblib.load(model_path)

def predict_price(sqft, bedrooms, bathrooms):
    model = load_model()
    data = np.array([[sqft, bedrooms, bathrooms]])
    prediction = model.predict(data)
    return prediction[0]


if __name__ == "__main__":
    price = predict_price(2000, 3, 2)
    print("Predicted Price:", price)