import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

LOOKBACK = 12


def create_sequences(X, lookback=12):
    X_seq = []
    for i in range(lookback, len(X)):
        X_seq.append(X[(i - lookback + 1):(i + 1)])
    return np.array(X_seq)


def main():
    # Load artifacts
    model = load_model("../models/trained_model.h5", compile=False)
    scaler_X = joblib.load("../models/scaler_X.pkl")
    scaler_y = joblib.load("../models/scaler_y.pkl")
    selected_features = joblib.load("../models/selected_features.pkl")

    # load already prepared input data
    # This file must contain the same selected feature columns
    input_df = pd.read_csv("../data/processed/selected_features_data.csv")

    # Keep only selected features
    X_input = input_df[selected_features].copy()

    # Scale
    X_scaled = scaler_X.transform(X_input)

    # Create sequences
    X_seq = create_sequences(X_scaled, lookback=LOOKBACK)

    # Predict
    pred_scaled = model.predict(X_seq, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled).flatten()

    print("Model successfully loaded.")
    print("Predictions:")
    print(pred[:10])

    actual = input_df["Appliances"].values[LOOKBACK:]

    print("\nActual vs Predicted:")
    n = min(11, len(pred), len(actual))
    for i in range(1, n):   
        print(f"Actual: {actual[i]:.2f} | Predicted: {pred[i]:.2f}")


if __name__ == "__main__":
    main()