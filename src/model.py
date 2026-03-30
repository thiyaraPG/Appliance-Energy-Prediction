import itertools
import random

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    Activation,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    Multiply,
    Permute,
    RepeatVector,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error


def create_sequences(X, y, lookback=6):
    """
    Create sequential data for deep learning models.
    Each sample uses the current row and the previous (lookback-1) rows
    """
    X_seq, y_seq = [], []

    for i in range(lookback, len(X)):
        X_seq.append(X[(i - lookback + 1):(i + 1)])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


def prepare_sequence_data(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, lookback=12):
    """
    Prepare train and test sequences for deep learning models.
    """
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, lookback)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


def split_train_validation_sequences(X_train_seq, y_train_seq):
    """A validation set was created by splitting 10% of the 
    training data (8% of the total dataset) to monitor model 
    performance and prevent overfitting.
    """
    val_size = int(0.1 * len(X_train_seq))

    X_tr = X_train_seq[:-val_size]
    y_tr = y_train_seq[:-val_size]

    X_val = X_train_seq[-val_size:]
    y_val = y_train_seq[-val_size:]

    return X_tr, y_tr, X_val, y_val


def build_model_a(X_train_seq):
    """
    Build baseline LSTM Model A
    """
    model = Sequential([
        LSTM(32, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dense(1)
    ], name="LSTM_Model_A")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mae",
        metrics=["mae"]
    )
    return model


def build_model_b(X_train_seq):
    """
    Build enhanced LSTM Model B
    """
    model = Sequential([
        LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="LSTM_Model_B")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mae",
        metrics=["mae"]
    )
    return model


def build_model_c(X_train_seq):
    """
    Build CNN + BiLSTM + Attention Model C
    """
    inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

    x = Conv1D(filters=32, kernel_size=2, activation="relu")(inputs)
    x = Dropout(0.2)(x)

    lstm_out = Bidirectional(LSTM(64, activation="tanh", return_sequences=True))(x)

    e = Dense(1, activation="tanh")(lstm_out)
    e = Flatten()(e)
    a = Activation("softmax")(e)

    a = RepeatVector(128)(a)  # 128 = 2 * 64 units
    a = Permute([2, 1])(a)
    output_attention = Multiply()([lstm_out, a])

    context_vector = Lambda(lambda z: K.sum(z, axis=1))(output_attention)

    x = Dense(32, activation="relu")(context_vector)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTM_Model_C")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mae",
        metrics=["mae"]
    )
    return model


def build_model_d(X_train_seq):
    """
    Build GRU Model D
    """
    model = Sequential([
        GRU(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.1),
        Dense(32, activation="relu"),
        Dense(1)
    ], name="GRU_Model_D")

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mae",
        metrics=["mae"]
    )
    return model


def train_initial_models(LSTM_Model_A, LSTM_Model_B, LSTM_Model_C, GRU_Model_D, X_tr, y_tr, X_val, y_val):
    """
    Train all four initial deep learning models
    """
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

    EPOCHS = 100
    BATCH_SIZE = 64

    print("Training LSTM_Model_A")
    history_A = LSTM_Model_A.fit(
        X_tr, y_tr,
        epochs=10,
        batch_size=256,
        validation_data=(X_val, y_val),
        verbose=0,
        shuffle=False
    )

    print("\nTraining LSTM_Model_B")
    history_B = LSTM_Model_B.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
        shuffle=False
    )

    print("\nTraining LSTM_Model_C")
    history_C = LSTM_Model_C.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1,
        shuffle=False
    )

    print("\nTraining GRU_Model_D")
    history_D = GRU_Model_D.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
        shuffle=False
    )

    return history_A, history_B, history_C, history_D


def build_lstm_b_candidate(X_train_seq, units, n_layers, lr, dropout_rate):
    """
    Build candidate model for focused LSTM_Model_B hyperparameter optimization
    """
    model = Sequential(name="Optimized_LSTM_Model_B")

    if n_layers == 1:
        model.add(
            LSTM(
                units,
                input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                kernel_regularizer=l2(0.001)
            )
        )
        model.add(Dropout(dropout_rate))

    elif n_layers == 2:
        model.add(
            LSTM(
                units,
                return_sequences=True,
                input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                kernel_regularizer=l2(0.001)
            )
        )
        model.add(Dropout(dropout_rate))

        model.add(
            LSTM(
                units // 2,
                kernel_regularizer=l2(0.001)
            )
        )
        model.add(Dropout(dropout_rate))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mae",
        metrics=["mae"]
    )
    return model


def run_hyperparameter_search(X_train_seq, X_tr, y_tr, X_val, y_val, scaler_y, best_model_name):
    """
    Run focused hyperparameter optimization for LSTM_Model_B only
    """
    print(f"Optimizing {best_model_name}")

    if best_model_name != "LSTM_Model_B":
        print(f"Warning: best_model_name is {best_model_name}, but this tuner is designed for LSTM_Model_B only.")

    param_grid = {
        "units": [64, 128],
        "n_layers": [1, 2],
        "lr": [0.001, 0.0005],
        "dropout_rate": [0.1, 0.2, 0.3],
        "batch_size": [16, 32]
    }

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    random.seed(42)
    sampled_params = random.sample(all_combinations, min(len(all_combinations), 10))

    baseline_p = {
        "units": 64,
        "n_layers": 1,
        "lr": 0.001,
        "dropout_rate": 0.1,
        "batch_size": 32
    }
    if baseline_p not in sampled_params:
        sampled_params.append(baseline_p)

    tuning_results = []

    early_stop_tuning = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )

    for i, params in enumerate(sampled_params):
        print(f"\nTesting Candidate {i+1}/{len(sampled_params)}: {params}")

        cand_model = build_lstm_b_candidate(
            X_train_seq=X_train_seq,
            units=params["units"],
            n_layers=params["n_layers"],
            lr=params["lr"],
            dropout_rate=params["dropout_rate"]
        )

        cand_model.fit(
            X_tr, y_tr,
            epochs=30,
            batch_size=params["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=[early_stop_tuning],
            verbose=0,
            shuffle=False
        )

        val_pred_scaled = cand_model.predict(X_val, verbose=0)
        val_pred = scaler_y.inverse_transform(val_pred_scaled.reshape(-1, 1)).flatten()
        val_true = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

        val_mae = mean_absolute_error(val_true, val_pred)
        val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))

        tuning_results.append({
            **params,
            "val_mae": val_mae,
            "val_rmse": val_rmse
        })

    search_df = pd.DataFrame(tuning_results).sort_values(by=["val_mae", "val_rmse"]).reset_index(drop=True)
    return search_df


def train_optimized_model(X_train_seq, search_df, best_model_name, X_tr, y_tr, X_val, y_val):
    """
    Build and train the final optimized LSTM_Model_B
    """
    best_p = search_df.iloc[0].to_dict()

    optimized_model = build_lstm_b_candidate(
        X_train_seq=X_train_seq,
        units=int(best_p["units"]),
        n_layers=int(best_p["n_layers"]),
        lr=float(best_p["lr"]),
        dropout_rate=float(best_p["dropout_rate"])
    )
    optimized_model._name = "Optimized_LSTM_Model_B"

    early_stop_opt = EarlyStopping(
        monitor="val_loss",
        patience=35,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_opt = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )

    print("Training Optimized LSTM_Model_B")
    history_opt = optimized_model.fit(
        X_tr, y_tr,
        epochs=300,
        batch_size=int(best_p["batch_size"]),
        validation_data=(X_val, y_val),
        callbacks=[early_stop_opt, reduce_lr_opt],
        verbose=1,
        shuffle=False
    )

    return optimized_model, history_opt, best_p