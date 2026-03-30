import os
import random
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def load_dataset(file_path="data/raw/energy_data_set.csv"):
    """
    Load dataset from CSV.
    """
    df = pd.read_csv(file_path)
    return df


def print_basic_info(df):
    """
    Print basic dataset information.
    """
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print(df.describe())


def preprocess_date(df):
    """
    Convert date column to datetime and sort dataset.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(df["date"].dtype)
    return df


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def split_train_test(df_clean):
    """
    Split cleaned dataframe into train and test sets.
    """
    split_index = int(0.8 * len(df_clean))

    train_df = df_clean.iloc[:split_index].copy()
    test_df = df_clean.iloc[split_index:].copy()

    X_train_full = train_df.drop(columns=["date", "Appliances"])
    y_train = train_df["Appliances"].copy()

    X_test_full = test_df.drop(columns=["date", "Appliances"])
    y_test = test_df["Appliances"].copy()

    print("Train shape:", X_train_full.shape, y_train.shape)
    print("Test shape:", X_test_full.shape, y_test.shape)

    return train_df, test_df, X_train_full, y_train, X_test_full, y_test


def handle_target_outliers(y_train, y_test):
    """
    Handle outliers in target variable using IQR on training data only.
    """
    Q1 = y_train.quantile(0.25)
    Q3 = y_train.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    y_outliers_before = ((y_train < lower_bound) | (y_train > upper_bound)).sum()

    print("Number of outliers in y_train before handling:", y_outliers_before)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)

    y_train = np.clip(y_train, lower_bound, upper_bound)
    y_test = np.clip(y_test, lower_bound, upper_bound)

    return y_train, y_test, lower_bound, upper_bound


def handle_feature_outliers(X_train_full, X_test_full):
    """
    Handle outliers in feature columns using training data only.
    """
    X_train_full = X_train_full.copy()
    X_test_full = X_test_full.copy()

    feature_bounds = {}

    print("Outlier count in each X_train_full feature before and after handling:\n")

    for col in X_train_full.columns:
        Q1 = X_train_full[col].quantile(0.25)
        Q3 = X_train_full[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outlier_count_before = ((X_train_full[col] < lower) | (X_train_full[col] > upper)).sum()

        feature_bounds[col] = (lower, upper)

        X_train_full[col] = X_train_full[col].clip(lower, upper)
        X_test_full[col] = X_test_full[col].clip(lower, upper)

        outlier_count_after = ((X_train_full[col] < lower) | (X_train_full[col] > upper)).sum()

        print(f"{col}: before = {outlier_count_before}, after = {outlier_count_after}")

    return X_train_full, X_test_full, feature_bounds


def scale_data(X_train, X_test, X_train_full, X_test_full, y_train, y_test):
    """
    Scale selected features, full features, and target variable.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_X_full = MinMaxScaler()
    X_train_full_scaled = scaler_X_full.fit_transform(X_train_full)
    X_test_full_scaled = scaler_X_full.transform(X_test_full)

    y_train_scaled = scaler_y.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test_scaled = scaler_y.transform(np.array(y_test).reshape(-1, 1))

    return (
        scaler_X,
        scaler_y,
        scaler_X_full,
        X_train_scaled,
        X_test_scaled,
        X_train_full_scaled,
        X_test_full_scaled,
        y_train_scaled,
        y_test_scaled,
    )

def save_clean_data(df_clean, output_path="processed/energy_data_clean.csv"):
    """
    Save cleaned dataset to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def save_train_test_data(train_df, test_df, train_path="processed/train_data.csv", test_path="processed/test_data.csv"):
    """
    Save train and test datasets to CSV.
    """
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")