import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import os

# Path
FEATURES_ELO_PATH = Path("data/processed/features_matches_long_elo_22_23.csv")
CALIBRATION_PLOT_ELO_PATH = Path("results/calibration_plot_elo.png")

# Load long-format match features already enriched with Elo (two rows per match, one per team).
def load_long_features_with_elo(path: str = FEATURES_ELO_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  -> Loaded long features (with ELO): {df.shape}")
    return df

# Drop rows with NaNs in selected features or target, sort by date and return feature matrix X and target vector y.
def prepare_dataset(df: pd.DataFrame, feature_cols: list[str], target_col: str = "result") -> tuple[pd.DataFrame, pd.Series]:
    initial_shape = df.shape
    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    # Ensure date is datetime and sort chronologically
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    return X, y

# Chronological train/test split (no shuffling) to avoid leakage. The first `train_ratio` fraction of matches are used for training,the remaining for testing.
def time_based_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_size = int(train_ratio * len(X))

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    print(f"  -> Train size: {len(X_train)} | Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

# Encode match results ('H', 'D', 'A') as integer labels.
def encode_target(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"  -> Target encoded. Classes: {le.classes_}")

    return y_train_enc, y_test_enc, le

# Standardize features based on TRAIN set only.
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("  -> Features standardized.")
    return X_train_scaled, X_test_scaled, scaler

# Train multinomial logistic regression.
def train_logistic_model(X_train_scaled: np.ndarray, y_train_enc: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train_enc)
    return model
 
# Compute accuracy, log loss and Brier scores for a fitted model. Also returns predicted probabilities and encoded predictions.
def evaluate_model(model: LogisticRegression, X_test_scaled: np.ndarray, y_test_enc: np.ndarray, le: LabelEncoder) -> dict:
    # Step 1: predictions
    y_proba = model.predict_proba(X_test_scaled)
    y_pred_enc = model.predict(X_test_scaled)

    # Step 2: accuracy
    accuracy = accuracy_score(y_test_enc, y_pred_enc)

    # Step 3: log loss
    ll = log_loss(y_test_enc, y_proba)

    # Step 4: Brier scores per class
    brier_scores = []

    for class_idx, class_label in enumerate(le.classes_):
        y_true_binary = (y_test_enc == class_idx).astype(int)
        y_prob_class = y_proba[:, class_idx]
        brier = brier_score_loss(y_true_binary, y_prob_class)
        brier_scores.append(brier)

    mean_brier = float(np.mean(brier_scores))

    # step 5: Results
    print(" --- Model Evaluation (with ELO) (Test Set) ---")
    print(f"  -> Accuracy: {accuracy:.3f}")
    print(f"  -> Log Loss: {ll:.4f}")
    print(f"  -> Mean Brier score: {mean_brier:.4f}")

    return {
        "y_proba": y_proba,
        "y_pred_enc": y_pred_enc,
        "accuracy": float(accuracy),
        "log_loss": float(ll),
        "brier_scores": brier_scores,
        "mean_brier": mean_brier,
    }

# Plot reliability / calibration curves for each class.
def plot_calibration_curves(y_test_enc: np.ndarray, y_proba: np.ndarray, le: LabelEncoder, save_path: Path=CALIBRATION_PLOT_ELO_PATH, n_bins: int = 10) -> None:
    os.makedirs(save_path.parent, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for class_idx, class_label in enumerate(le.classes_):
        # Binary labels: 1 if this class, 0 otherwise
        y_true_binary = (y_test_enc == class_idx).astype(int)
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
            linestyle = "-",
            label=f"Class {class_label}",
        )

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency (True probability)")
    plt.title("Calibration Curve: Logistic Regression + Elo")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 
    print(f"  -> Calibration plot saved to: {save_path}")

# Main function to run the full Elo modelling pipeline: load, preprocess, train Logistic Regression, evaluate, and plot calibration.
def run_baseline_with_elo(features_path: Path = FEATURES_ELO_PATH, plot_path: Path = CALIBRATION_PLOT_ELO_PATH, train_ratio: float = 0.8) -> tuple[LogisticRegression, dict]:
    print("\n===== Starting Logistic Model Training - WITH ELO (08) =====")

    # --- Configuration ---
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
    # 1. Load data
    df = load_long_features_with_elo(features_path)

    # 2. Prepare dataset
    X, y = prepare_dataset(df, feature_cols, target_col="result")

    # 3. Time-based split
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=train_ratio)

    # 4. Preprocessing
    y_train_enc, y_test_enc, le = encode_target(y_train, y_test)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 5. Train model
    print("  -> Training Logistic Regression model (with ELO)...")
    model = train_logistic_model(X_train_scaled, y_train_enc)
    print("  -> Model trained successfully.")

    # 6. Evaluate Model
    print("  -> Evaluating model on test set...")
    metrics = evaluate_model(model, X_test_scaled, y_test_enc, le)
    
    # 7. Save calibration curves
    print("  -> Creating and saving calibration curve plot.")
    plot_calibration_curves(
        y_test_enc=y_test_enc,
        y_proba=metrics["y_proba"],
        le=le,
        n_bins=10,
        save_path=plot_path
    )

    print("===== Logistic Model Training (08) Complete. ✅ =====")
    return model, metrics

if __name__ == "__main__":
    run_baseline_with_elo()

