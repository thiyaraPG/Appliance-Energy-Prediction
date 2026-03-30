import os
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


def get_time_of_day(hour):
    """
    Convert hour into time-of-day category.
    """
    if 6 <= hour < 12:
        return 0
    elif 12 <= hour < 18:
        return 1
    elif 18 <= hour < 22:
        return 2
    else:
        return 3


def add_time_features(df):
    """
    Add time-based features.
    """
    df = df.copy()

    df["hour"] = df["date"].dt.hour
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    df["time_of_day"] = df["hour"].apply(get_time_of_day)

    df["is_morning_peak"] = df["hour"].isin([7, 8, 9]).astype(int)
    df["is_evening_peak"] = df["hour"].isin([18, 19, 20, 21]).astype(int)
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

    return df


def add_lag_features(df):
    """
    Add lagged target features.
    """
    df = df.copy()

    df["lag_1"] = df["Appliances"].shift(1)
    df["lag_3"] = df["Appliances"].shift(3)
    df["lag_6"] = df["Appliances"].shift(6)
    df["lag_12"] = df["Appliances"].shift(12)
    df["lag_144"] = df["Appliances"].shift(144)

    return df


def add_rolling_features(df):
    """
    Add rolling window statistics.
    """
    df = df.copy()

    df["rolling_mean_1h"] = df["Appliances"].shift(1).rolling(window=6).mean()
    df["rolling_mean_3h"] = df["Appliances"].shift(1).rolling(window=18).mean()
    df["rolling_std_1h"] = df["Appliances"].shift(1).rolling(window=6).std()

    return df


def add_interaction_features(df):
    """
    Add interaction features.
    """
    df = df.copy()

    df["T_indoor_avg"] = df[["T1", "T2", "T3", "T4", "T5", "T6"]].mean(axis=1)
    df["T_diff"] = df["T_indoor_avg"] - df["T_out"]

    df["RH_indoor_avg"] = df[["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6"]].mean(axis=1)
    df["RH_diff"] = df["RH_indoor_avg"] - df["RH_out"]

    df["T1_RH1"] = df["T1"] * df["RH_1"]
    df["T_out_RH_out"] = df["T_out"] * df["RH_out"]
    df["lights_hour"] = df["lights"] * df["hour"]

    return df


def remove_noise_and_dropna(df):
    """
    Remove known noise columns and drop NaN rows created by lag/rolling features.
    """
    df = df.copy()

    drop_noise = [col for col in ["rv1", "rv2"] if col in df.columns]
    df = df.drop(columns=drop_noise)

    df_clean = df.dropna().reset_index(drop=True)
    return df_clean


def select_features_with_corr_and_rfe(X_train_full, y_train, X_test_full, top_n_corr=20, n_features_to_select=10):
    """
    Perform correlation analysis and RFE on training data only.
    """
    corr_with_target = X_train_full.corrwith(y_train).abs().sort_values(ascending=False)

    print("Top 20 features by correlation with target:")
    print(corr_with_target.head(20))

    corr_features = corr_with_target.head(top_n_corr).index.tolist()

    print("\nTop correlated features:")
    print(corr_features)

    X_train_rfe = X_train_full[corr_features]
    X_test_rfe = X_test_full[corr_features]

    rfe_model = LinearRegression()
    rfe = RFE(estimator=rfe_model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train_rfe, y_train)

    selected_features = X_train_rfe.columns[rfe.support_].tolist()

    print("\nFinal selected features (Correlation + RFE):")
    print(selected_features)

    X_train = X_train_full[selected_features].copy()
    X_test = X_test_full[selected_features].copy()

    print("\nFinal selected train shape:", X_train.shape)
    print("Final selected test shape:", X_test.shape)

    return selected_features, X_train, X_test

def save_selected_features_data(df_clean, selected_features, output_path="processed/selected_features_data.csv"):
    """
    Save dataset with date, target, and final selected features.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    final_selected_features_data = df_clean[["date", "Appliances"] + selected_features].copy()
    final_selected_features_data.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    return final_selected_features_data