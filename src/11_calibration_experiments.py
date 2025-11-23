import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Paths
FEATURES_ELO_PATH = "data/processed/features_matches_long_elo_22_23.csv"
CALIB_SUMMARY_PATH = "results/calibration_summary_22_23.csv"


# Load long-format match features already enriched with Elo (two rows per match, one per team).
def load_features_with_elo(path: str = FEATURES_ELO_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded long features:{df.shape}")
    return df

# Drop rows with NaNs in selected features or target, sort by date and return feature matrix X and target vector y.
def prepare_dataset(df: pd.DataFrame, feature_cols: list[str], target_col: str = "result") -> tuple[pd.DataFrame, pd.Series]:

    print("\nShape BEFORE dropna:", df.shape)
    print("NaN per selected column BEFORE drop:")
    print(df[feature_cols + [target_col]].isna().sum())

    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    print("\nShape AFTER dropna:", df_clean.shape)
    print("NaN per selected column AFTER drop:")
    print(df_clean[feature_cols + [target_col]].isna().sum())

    df_clean["date"] = pd.to_datetime(df_clean["date"])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    return X, y

# Chronological train/test split (no shuffling) to avoid leakage. The first `train_ratio` fraction of observations are used for training, the remaining ones for testing.
def time_based_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_size = int(train_ratio * len(X))

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    print("\nTrain shapes:", X_train.shape, y_train.shape)
    print("Test shapes :", X_test.shape, y_test.shape)

    print("\nNaN in TRAIN features:")
    print(X_train.isna().sum())

    print("\nNaN in TEST features:")
    print(X_test.isna().sum())

    return X_train, X_test, y_train, y_test

# Encode match results ('H', 'D', 'A') as integer labels.
def encode_target(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("\nClasses:", le.classes_)  
    print("First 10 encoded y_train:", y_train_enc[:10])

    return y_train_enc, y_test_enc, le

# Standardize features based on TRAIN set only.
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTrain scaled shape:", X_train_scaled.shape)
    print("Test scaled shape :", X_test_scaled.shape)

    print("NaN in X_train_scaled:", np.isnan(X_train_scaled).sum())
    print("NaN in X_test_scaled :", np.isnan(X_test_scaled).sum())

    return X_train_scaled, X_test_scaled, scaler

# Construct base classifiers (without calibration). Returns a dict {model_name: estimator}.
def build_base_models(random_state: int = 42) -> dict[str, object]:
    models: dict[str, object] = {}

    # Multinomial logistic regression
    models["Logistic"] = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
        random_state=random_state,
    )

    # Random Forest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=1,
    )

    # XGBoost
    models["XGBoost"] = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    return models

# Compute Brier score per class and mean Brier score from multiclass probabilities.
def compute_brier_scores(y_true_enc: np.ndarray, y_proba: np.ndarray, classes: np.ndarray) -> tuple[list[float], float]:
    brier_scores: list[float] = []

    for class_idx, _ in enumerate(classes):
        y_true_binary = (y_true_enc == class_idx).astype(int)
        y_prob_class = y_proba[:, class_idx]
        brier = brier_score_loss(y_true_binary, y_prob_class)
        brier_scores.append(float(brier))

    mean_brier = float(np.mean(brier_scores))
    return brier_scores, mean_brier

# Package all metrics for a given (model, calibration) pair.
def evaluate_predictions(model_name: str, calibration: str, y_true_enc: np.ndarray, y_proba: np.ndarray, y_pred_enc: np.ndarray, classes: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true_enc, y_pred_enc))
    ll = float(log_loss(y_true_enc, y_proba))
    brier_scores, mean_brier = compute_brier_scores(y_true_enc, y_proba, classes)

    print(
        f"\n[{model_name} | {calibration}] "
        f"Accuracy={acc:.3f}  LogLoss={ll:.4f}  MeanBrier={mean_brier:.4f}"
    )

    return {
        "model": model_name,
        "calibration": calibration,
        "accuracy": acc,
        "log_loss": ll,
        "mean_brier": mean_brier,
        "brier_scores": brier_scores,
        "y_proba": y_proba,
        "y_pred_enc": y_pred_enc,
    }

# Save summary metrics to CSV in the results folder.
def save_calibration_summary(summary: pd.DataFrame, path: str = CALIB_SUMMARY_PATH) -> None:
    summary.to_csv(path, index=False)
    print(f"\nCalibration summary saved to: {path}")

# Plot multiclass calibration curves (one curve per class).
def plot_multiclass_calibration(y_true_enc: np.ndarray, y_proba: np.ndarray, classes: np.ndarray, n_bins: int = 10, title: str = "Calibration curves") -> None:
    plt.figure(figsize=(8, 6))

    for class_idx, class_label in enumerate(classes):
        # Binary labels: 1 if this class, 0 otherwise
        y_true_binary = (y_true_enc == class_idx).astype(int)
        y_prob_class = y_proba[:, class_idx]

        # Compute calibration curve
        true_frac, pred_mean = calibration_curve(
            y_true_binary,
            y_prob_class,
            n_bins=n_bins,
            strategy="uniform",
        )

        plt.plot(
            pred_mean,
            true_frac,
            marker="o",
            label=f"Class {class_label}",
        )

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Select the best (model, calibration) by lowest log loss and plot its calibration curves.
def plot_best_calibration(results: list[dict], y_true_enc: np.ndarray, classes: np.ndarray) -> None:
    best = min(results, key=lambda r: r["log_loss"])
    title = f"Calibration curves ({best['model']} + {best['calibration']})"

    print(
        f"\nBest combination by log loss: "
        f"{best['model']} | {best['calibration']} "
        f"(log_loss={best['log_loss']:.4f})"
    )

    plot_multiclass_calibration(
        y_true_enc=y_true_enc,
        y_proba=best["y_proba"],
        classes=classes,
        n_bins=10,
        title=title,
    )


# Main function: run raw / Platt / Isotonic calibration for Logistic, RandomForest and XGBoost.
# Returns: 
    # summary  : DataFrame with metrics 
    # results  : list of dicts including probabilities and metrics
    # y_test   : encoded test labels  
    # classes  : label encoder classes (['A', 'D', 'H'])
def run_calibration_experiments() -> tuple[pd.DataFrame, list[dict], np.ndarray, np.ndarray]:
    # 1) Load data
    df = load_features_with_elo()

    # 2) Choose features and target
    feature_cols = [
        "is_home",
        "rolling_xg_for_5",
        "rolling_xg_against_5",
        "rolling_xg_diff_5",
        "rolling_points_5",
        "strength_points_5",
        "elo_home_before",
        "elo_away_before",
        "elo_diff_home",
    ]

    X, y = prepare_dataset(df, feature_cols, target_col="result")

    # 3) Time-based split
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=0.8)

    # 4) Encode target
    y_train_enc, y_test_enc, le = encode_target(y_train, y_test)

    # 5) Scale features (same scaled inputs for all models)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 6) Build base models
    base_models = build_base_models(random_state=42)

    # 7) For each model, evaluate:
        # raw probabilities
        # Platt scaling (sigmoid)
        #Isotonic regression
    results: list[dict] = []

    for model_name, base_estimator in base_models.items():
        print("\n" + "=" * 70)
        print(f"Model: {model_name}")
        print("=" * 70)

        # --- Raw (no calibration) ---
        raw_model = base_estimator
        raw_model.fit(X_train_scaled, y_train_enc)

        y_proba_raw = raw_model.predict_proba(X_test_scaled)
        y_pred_raw = raw_model.predict(X_test_scaled)

        res_raw = evaluate_predictions(
            model_name=model_name,
            calibration="raw",
            y_true_enc=y_test_enc,
            y_proba=y_proba_raw,
            y_pred_enc=y_pred_raw,
            classes=le.classes_,
        )
        results.append(res_raw)

        # --- Platt scaling (sigmoid) ---
        platt = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=3,
        )
        platt.fit(X_train_scaled, y_train_enc)

        y_proba_platt = platt.predict_proba(X_test_scaled)
        y_pred_platt = platt.predict(X_test_scaled)

        res_platt = evaluate_predictions(
            model_name=model_name,
            calibration="Platt",
            y_true_enc=y_test_enc,
            y_proba=y_proba_platt,
            y_pred_enc=y_pred_platt,
            classes=le.classes_,
        )
        results.append(res_platt)

        # --- Isotonic regression ---
        iso = CalibratedClassifierCV(
            estimator=base_estimator,
            method="isotonic",
            cv=3,
        )
        iso.fit(X_train_scaled, y_train_enc)

        y_proba_iso = iso.predict_proba(X_test_scaled)
        y_pred_iso = iso.predict(X_test_scaled)

        res_iso = evaluate_predictions(
            model_name=model_name,
            calibration="Isotonic",
            y_true_enc=y_test_enc,
            y_proba=y_proba_iso,
            y_pred_enc=y_pred_iso,
            classes=le.classes_,
        )
        results.append(res_iso)

    # 8) Build summary DataFrame
    summary = pd.DataFrame(
        [
            {
                "model": r["model"],
                "calibration": r["calibration"],
                "accuracy": r["accuracy"],
                "log_loss": r["log_loss"],
                "mean_brier": r["mean_brier"],
            }
            for r in results
        ]
    )

    # Sort for nicer display
    summary = summary.sort_values(["model", "calibration"]).reset_index(drop=True)

    print("\n\n=== Calibration summary ===")
    print(summary)

    return summary, results, y_test_enc, le.classes_

if __name__ == "__main__":
    summary_df, all_results, y_test_enc, classes = run_calibration_experiments()

    # Save to CSV
    save_calibration_summary(summary_df, CALIB_SUMMARY_PATH)

    # Plot the best combination by log loss
    plot_best_calibration(all_results, y_test_enc, classes)