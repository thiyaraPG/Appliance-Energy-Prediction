import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf

from data_preprocessing import (
    handle_feature_outliers,
    handle_target_outliers,
    load_dataset,
    preprocess_date,
    print_basic_info,
    scale_data,
    set_random_seed,
    split_train_test,
    save_clean_data,
    save_train_test_data,
)
from evaluation import (
    build_baseline_results,
    build_optimized_comparison,
    evaluate_all_initial_dl_models,
    evaluate_linear_regression,
    evaluate_optimized_model,
    evaluate_random_forest,
)
from feature_engineering import (
    add_interaction_features,
    add_lag_features,
    add_rolling_features,
    add_time_features,
    remove_noise_and_dropna,
    select_features_with_corr_and_rfe,
    save_selected_features_data,
)
from model import (
    build_model_a,
    build_model_b,
    build_model_c,
    build_model_d,
    prepare_sequence_data,
    run_hyperparameter_search,
    split_train_validation_sequences,
    train_initial_models,
    train_optimized_model,
)


def main():

    # 1. DATA UNDERSTANDING

    df = load_dataset("../data/raw/energy_data_set.csv")
    print_basic_info(df)

    df = preprocess_date(df)

    # EDA plots
    plt.figure(figsize=(15, 5))
    plt.plot(df["date"], df["Appliances"])
    plt.title("Energy consumption over time")
    plt.xlabel("Date")
    plt.ylabel("Appliances (Wh)")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.histplot(df["Appliances"], bins=50, kde=True)
    plt.title("Distribution of appliances energy")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df["Appliances"])
    plt.title("Boxplot of appliances")
    plt.show()

    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation heatmap")
    plt.show()

    df["hour"] = df["date"].dt.hour
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="hour", y="Appliances", data=df)
    plt.title("Energy usage by Hour")
    plt.show()

    # 2. FEATURE ENGINEERING

    df = add_time_features(df)

    hourly_avg = df.groupby("hour")["Appliances"].mean()
    plt.figure(figsize=(12, 4))
    plt.bar(hourly_avg.index, hourly_avg.values, color="steelblue")
    plt.title("Average Energy Consumption by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Avg Appliances (Wh)")
    plt.xticks(range(0, 24))
    plt.grid(axis="y", alpha=0.3)
    plt.show()

    plot_acf(df["Appliances"].dropna(), lags=50)
    plt.title("Autocorrelation of appliances energy")
    plt.show()

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    df_clean = remove_noise_and_dropna(df)

    save_clean_data(df_clean)

    # 3. PREPROCESSING

    set_random_seed(42)

    train_df, test_df, X_train_full, y_train, X_test_full, y_test = split_train_test(df_clean)
    save_train_test_data(train_df, test_df)

    y_train, y_test, lower_bound, upper_bound = handle_target_outliers(y_train, y_test)

    plt.figure()
    plt.boxplot(y_train)
    plt.title("Boxplot of appliances after outlier handling")
    plt.ylabel("Appliances")
    plt.show()

    X_train_full, X_test_full, feature_bounds = handle_feature_outliers(X_train_full, X_test_full)

    selected_features, X_train, X_test = select_features_with_corr_and_rfe(
        X_train_full, y_train, X_test_full, top_n_corr=20, n_features_to_select=10
    )
    save_selected_features_data(df_clean, selected_features)

    (
        scaler_X,
        scaler_y,
        scaler_X_full,
        X_train_scaled,
        X_test_scaled,
        X_train_full_scaled,
        X_test_full_scaled,
        y_train_scaled,
        y_test_scaled,
    ) = scale_data(X_train, X_test, X_train_full, X_test_full, y_train, y_test)

    # 4. BASELINE MODELS

    # Linear Regression 
    lr_lookback = 48

    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_scaled)

    lr_results = evaluate_linear_regression(
        lr_model, X_test_scaled, y_test_scaled, scaler_y, lookback=lr_lookback
    )

    # Random Forest 
    rf_lookback = 12

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train_scaled.ravel())

    rf_results = evaluate_random_forest(
        rf_model, X_test_scaled, y_test_scaled, scaler_y, lookback=rf_lookback
    )

    baseline_results = build_baseline_results(lr_results, rf_results)

    # 5. DEEP LEARNING MODELS

    lookback = 12

    X_train_seq, y_train_seq, X_test_seq, y_test_seq = prepare_sequence_data(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, lookback=lookback
    )

    print("X_train_seq shape:", X_train_seq.shape)
    print("y_train_seq shape:", y_train_seq.shape)
    print("X_test_seq shape :", X_test_seq.shape)
    print("y_test_seq shape :", y_test_seq.shape)

    X_test_lstm = X_test_seq
    X_tr, y_tr, X_val, y_val = split_train_validation_sequences(X_train_seq, y_train_seq)

    LSTM_Model_A = build_model_a(X_train_seq)
    LSTM_Model_A.summary()

    LSTM_Model_B = build_model_b(X_train_seq)
    LSTM_Model_B.summary()

    LSTM_Model_C = build_model_c(X_train_seq)
    LSTM_Model_C.summary()

    GRU_Model_D = build_model_d(X_train_seq)
    GRU_Model_D.summary()

    history_A, history_B, history_C, history_D = train_initial_models(
        LSTM_Model_A, LSTM_Model_B, LSTM_Model_C, GRU_Model_D,
        X_tr, y_tr, X_val, y_val
    )

    # Plot training curves
    fig, axes = plt.subplots(1, 4, figsize=(24, 4))

    histories = [
        (history_A, "LSTM_Model_A"),
        (history_B, "LSTM_Model_B"),
        (history_C, "LSTM_Model_C"),
        (history_D, "GRU_Model_D")
    ]

    for ax, (history, name) in zip(axes, histories):
        ax.plot(history.history["loss"], label="Train Loss", color="steelblue", linewidth=2)
        ax.plot(history.history["val_loss"], label="Val Loss", color="darkorange", linewidth=2)
        ax.set_title(f"{name} Learning Curve")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MAE Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    initial_dl_models = [
        (LSTM_Model_A, "LSTM_Model_A"),
        (LSTM_Model_B, "LSTM_Model_B"),
        (LSTM_Model_C, "LSTM_Model_C"),
        (GRU_Model_D, "GRU_Model_D")
    ]

    dl_results_df, preds_dict, best_dl_model_row, best_model_name, results_df, y_true = evaluate_all_initial_dl_models(
        initial_dl_models, X_test_lstm, y_test_seq, scaler_y, y_train
    )

    print(dl_results_df.round(4))

    # Visualization 1: Predicted vs Actual
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    for i, (mod, name) in enumerate(initial_dl_models):
        pred = preds_dict[name]
        time_axis = np.arange(len(y_true))[:200]
        axes[i].plot(time_axis, y_true[:200], label="Actual", color="gray", linestyle="--")
        axes[i].plot(time_axis, pred[:200], label="Predicted", color="steelblue")
        axes[i].set_title(f"{name}: Predicted vs Actual")
        axes[i].set_ylabel("Energy (Wh)")
        axes[i].set_xlabel("Time Step")
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Visualization 2: Residual plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    axes = axes.flatten()

    for i, (mod, name) in enumerate(initial_dl_models):
        pred = preds_dict[name]
        residuals = y_true - pred
        axes[i].scatter(pred, residuals, alpha=0.3, color="coral")
        axes[i].axhline(y=0, color="black", linestyle="--")
        axes[i].set_title(f"{name}: Residuals")
        axes[i].set_xlabel("Predicted (Wh)")
        axes[i].set_ylabel("Residual (Wh)")
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Visualization 3: Bar Chart of Evaluation Metrics
    models = results_df["Model"].tolist()
    mae_vals = results_df["MAE (Wh)"].tolist()
    rmse_vals = results_df["RMSE (Wh)"].tolist()
    r2_vals = results_df["R2"].tolist()

    x = np.arange(len(models))
    colors = ["gray", "gray", "lightcoral", "steelblue", "seagreen", "darkorange"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].bar(x, mae_vals, width=0.5, color=colors)
    axes[0].set_title("MAE (lower = better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=30, ha="right")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, rmse_vals, width=0.5, color=colors)
    axes[1].set_title("RMSE (lower = better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(x, r2_vals, width=0.5, color=colors)
    axes[2].set_title("R2 (higher = better)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=30, ha="right")
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 6. MODEL OPTIMIZATION

    search_df = run_hyperparameter_search(
        X_train_seq, X_tr, y_tr, X_val, y_val, scaler_y, best_model_name
    )
    print(search_df)

    best_params = search_df.iloc[0].to_dict()
    print("\nBest Hyperparameters:", best_params)

    optimized_model, history_opt, best_p = train_optimized_model(
        X_train_seq, search_df, best_model_name, X_tr, y_tr, X_val, y_val
    )

    pred_opt, y_test_orig_seq, mae_opt, rmse_opt, mape_opt, r2_opt = evaluate_optimized_model(
        optimized_model, X_test_lstm, y_test_seq, scaler_y
    )

    diff_df, best_row, mae_z_opt, rmse_z_opt = build_optimized_comparison(
        dl_results_df, best_model_name, mae_opt, rmse_opt, mape_opt, r2_opt, y_train
    )
    print(diff_df.round(4))

    # Final optimized vs baseline comparison 
    metrics_labels = ["MAE (Wh)", "RMSE (Wh)"]
    baseline_scores = [best_row["MAE (Wh)"], best_row["RMSE (Wh)"]]
    optimized_scores = [mae_opt, rmse_opt]

    x = np.arange(len(metrics_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width / 2, baseline_scores, width, label=f"Baseline ({best_model_name})", color="#3498db")
    rects2 = ax.bar(x + width / 2, optimized_scores, width, label="Optimized model", color="#e67e22")

    ax.set_ylabel("Score (Lower is Better)")
    ax.set_title("Final Model Performance Comparison (Error Metrics)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom"
            )

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.show()

    # Optimized model training history
    plt.figure(figsize=(10, 4))
    plt.plot(history_opt.history["loss"], label="Train Loss", alpha=0.9)
    if "val_loss" in history_opt.history:
        plt.plot(history_opt.history["val_loss"], label="Val Loss", alpha=0.9)

    plt.title("Optimized Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Final performance comparison
    best_init_row = results_df[results_df["Model"] == best_model_name].iloc[0]

    viz_models = ["Linear Regression", f"Best Initial ({best_model_name})", "Optimized model"]
    viz_mae = [lr_results["MAE (Wh)"], best_init_row["MAE (Wh)"], mae_opt]
    viz_rmse = [lr_results["RMSE (Wh)"], best_init_row["RMSE (Wh)"], rmse_opt]
    viz_r2 = [lr_results["R2"], best_init_row["R2"], r2_opt]

    x = np.arange(len(viz_models))
    colors = ["#95a5a6", "#3498db", "#e67e22"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        (viz_mae, "MAE comparison", "Lower is Better"),
        (viz_rmse, "RMSE comparison", "Lower is Better"),
        (viz_r2, "R2 comparison", "Higher is Better")
    ]

    for i, (data, title, sub) in enumerate(metrics):
        axes[i].bar(x, data, color=colors)
        axes[i].set_title(title)
        axes[i].set_xlabel(sub)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(viz_models, rotation=25, ha="right")
        axes[i].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Save model and preprocessing artifacts
    os.makedirs("../models", exist_ok=True)

    optimized_model.save("../models/trained_model.h5")
    joblib.dump(scaler_X, "../models/scaler_X.pkl")
    joblib.dump(scaler_y, "../models/scaler_y.pkl")
    joblib.dump(selected_features, "../models/selected_features.pkl")

    print("Model and preprocessing artifacts successfully saved.")


if __name__ == "__main__":
    main()