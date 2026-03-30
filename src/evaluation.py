import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error


def evaluate_linear_regression(lr_model, X_test_scaled, y_test_scaled, scaler_y, lookback=48):
    """
    Evaluate Linear Regression baseline model.
    Uses aligned comparison by trimming the first `lookback` test rows.
    """
    lr_pred_scaled = lr_model.predict(X_test_scaled)

    y_test_orig = scaler_y.inverse_transform(y_test_scaled).flatten()
    lr_pred = scaler_y.inverse_transform(lr_pred_scaled.reshape(-1, 1)).flatten()

    y_test_orig_aligned = y_test_orig[lookback:]
    lr_pred_aligned = lr_pred[lookback:]

    lr_mae = mean_absolute_error(y_test_orig_aligned, lr_pred_aligned)
    lr_rmse = np.sqrt(mean_squared_error(y_test_orig_aligned, lr_pred_aligned))
    lr_mape = mean_absolute_percentage_error(y_test_orig_aligned, lr_pred_aligned) * 100
    lr_r2 = r2_score(y_test_orig_aligned, lr_pred_aligned)

    print("Linear Regression")
    print(f"  MAE  : {lr_mae:.2f} Wh")
    print(f"  RMSE : {lr_rmse:.2f} Wh")
    print(f"  MAPE : {lr_mape:.2f} %")
    print(f"  R²   : {lr_r2:.4f}")

    return {
        "MAE (Wh)": lr_mae,
        "RMSE (Wh)": lr_rmse,
        "MAPE (%)": lr_mape,
        "R2": lr_r2,
        "y_true": y_test_orig_aligned,
        "y_pred": lr_pred_aligned
    }


def evaluate_random_forest(rf_model, X_test_scaled, y_test_scaled, scaler_y, lookback=12):
    """
    Evaluate Random Forest baseline model.
    Uses aligned comparison by trimming the first `lookback` test rows.
    """
    rf_pred_scaled = rf_model.predict(X_test_scaled)
    rf_pred = scaler_y.inverse_transform(rf_pred_scaled.reshape(-1, 1)).flatten()

    y_test_orig = scaler_y.inverse_transform(y_test_scaled).flatten()

    y_test_orig_aligned = y_test_orig[lookback:]
    rf_pred_aligned = rf_pred[lookback:]

    rf_mae = mean_absolute_error(y_test_orig_aligned, rf_pred_aligned)
    rf_rmse = np.sqrt(mean_squared_error(y_test_orig_aligned, rf_pred_aligned))
    rf_mape = mean_absolute_percentage_error(y_test_orig_aligned, rf_pred_aligned) * 100
    rf_r2 = r2_score(y_test_orig_aligned, rf_pred_aligned)

    print("Random Forest")
    print(f"  MAE  : {rf_mae:.2f} Wh")
    print(f"  RMSE : {rf_rmse:.2f} Wh")
    print(f"  MAPE : {rf_mape:.2f} %")
    print(f"  R²   : {rf_r2:.4f}")

    return {
        "MAE (Wh)": rf_mae,
        "RMSE (Wh)": rf_rmse,
        "MAPE (%)": rf_mape,
        "R2": rf_r2,
        "y_true": y_test_orig_aligned,
        "y_pred": rf_pred_aligned
    }


def build_baseline_results(lr_results, rf_results):
    """
    Build baseline results dataframe.
    """
    baseline_results = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "MAE (Wh)": [lr_results["MAE (Wh)"], rf_results["MAE (Wh)"]],
        "RMSE (Wh)": [lr_results["RMSE (Wh)"], rf_results["RMSE (Wh)"]],
        "MAPE (%)": [lr_results["MAPE (%)"], rf_results["MAPE (%)"]],
        "R2": [lr_results["R2"], rf_results["R2"]]
    })

    print("\nBaseline Summary")
    print(baseline_results.to_string(index=False))

    return baseline_results


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate a deep learning model on sequence-based test data.
    """
    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mape = mean_absolute_percentage_error(y_true, pred) * 100
    r2 = r2_score(y_true, pred)

    return pred, y_true, mae, rmse, mape, r2


def evaluate_all_initial_dl_models(initial_dl_models, X_test_lstm, y_test_seq, scaler_y, y_train):
    """
    Evaluate all initial deep learning models and select the best one by lowest MAE.
    """
    APPLIANCES_STD = y_train.std()

    preds_dict = {}
    dl_results = []

    for mod, name in initial_dl_models:
        pred, y_true, mae, rmse, mape, r2 = evaluate_model(mod, X_test_lstm, y_test_seq, scaler_y)
        preds_dict[name] = pred

        mae_z = mae / APPLIANCES_STD
        rmse_z = rmse / APPLIANCES_STD

        dl_results.append({
            "Model": name,
            "MAE (Wh)": mae,
            "RMSE (Wh)": rmse,
            "MAPE (%)": mape,
            "MAE (Z)": mae_z,
            "RMSE (Z)": rmse_z,
            "R2": r2
        })

    dl_results_df = pd.DataFrame(dl_results)
    best_dl_model_row = dl_results_df.loc[dl_results_df["MAE (Wh)"].idxmin()]
    best_model_name = best_dl_model_row["Model"]

    print(f"\nBEST MODEL SELECTED FOR OPTIMIZATION: {best_model_name}")

    results_df = dl_results_df.copy()

    return dl_results_df, preds_dict, best_dl_model_row, best_model_name, results_df, y_true


def evaluate_optimized_model(optimized_model, X_test_lstm, y_test_seq, scaler_y):
    """
    Evaluate optimized LSTM_Model_B on test data.
    """
    opt_pred_scaled = optimized_model.predict(X_test_lstm, verbose=0).flatten()
    pred_opt = scaler_y.inverse_transform(opt_pred_scaled.reshape(-1, 1)).flatten()
    y_test_orig_seq = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    mae_opt = mean_absolute_error(y_test_orig_seq, pred_opt)
    rmse_opt = np.sqrt(mean_squared_error(y_test_orig_seq, pred_opt))
    mape_opt = mean_absolute_percentage_error(y_test_orig_seq, pred_opt) * 100
    r2_opt = r2_score(y_test_orig_seq, pred_opt)

    print(f"Final MAE (Wh): {mae_opt:.2f}")
    print(f"Final RMSE (Wh): {rmse_opt:.2f}")
    print(f"Final MAPE (%): {mape_opt:.2f}")
    print(f"Final R²: {r2_opt:.4f}")

    return pred_opt, y_test_orig_seq, mae_opt, rmse_opt, mape_opt, r2_opt


def build_optimized_comparison(dl_results_df, best_model_name, mae_opt, rmse_opt, mape_opt, r2_opt, y_train):
    """
    Build comparison table for baseline best DL model vs optimized model.
    """
    APPLIANCES_STD = y_train.std()

    mae_z_opt = mae_opt / APPLIANCES_STD
    rmse_z_opt = rmse_opt / APPLIANCES_STD

    best_row = dl_results_df[dl_results_df["Model"] == best_model_name].iloc[0]

    print(f"\nFINAL REPORT: Optimized {best_model_name} Performance")
    print(f"  RAW MAE   : {mae_opt:.2f} Wh")
    print(f"  RAW RMSE  : {rmse_opt:.2f} Wh")
    print(f"  RAW MAPE  : {mape_opt:.2f}%")
    print(f"  R² SCORE  : {r2_opt:.4f}")

    diff_df = pd.DataFrame({
        "Stage": [f"Baseline ({best_model_name})", "Optimized model"],
        "MAE (Wh)": [best_row["MAE (Wh)"], mae_opt],
        "RMSE (Wh)": [best_row["RMSE (Wh)"], rmse_opt],
        "MAPE (%)": [best_row["MAPE (%)"], mape_opt],
        "R2": [best_row["R2"], r2_opt]
    })

    return diff_df, best_row, mae_z_opt, rmse_z_opt