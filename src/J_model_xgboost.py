import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import os

# Input path  
FEATURES_ELO_PATH = Path("data/processed/features_matches_long_elo_22_23.csv")
#Output path
CALIBRATION_PLOT_XGB_PATH = Path("results/calibration_plot_xgboost.png")

# Load long-format match features already enriched with Elo (two rows per match, one per team).
def load_long_features_with_elo(path: Path= FEATURES_ELO_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  -> Loaded long features : {df.shape}")
    return df

# Drop rows with NaNs in selected features or target, sort by date and return feature matrix X and target vector y.
def prepare_dataset(df: pd.DataFrame, feature_cols: list[str], target_col: str = "result") -> tuple[pd.DataFrame, pd.Series]:
    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    # Ensure date is datetime and sort chronologically
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    return X, y

# Chronological train/test split (no shuffling) to avoid leakage. The first `train_ratio` fraction of observations are used for training, the remaining ones for testing.
def time_based_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_size = int(train_ratio * len(X))

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    print(f"  -> Train size: {len(X_train)} | Test size: {len(X_test)} ")

    return X_train, X_test, y_train, y_test

# Encode match results ('H', 'D', 'A') as integer labels.
def encode_target(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"  -> Target encoded. Classes: {le.classes_}")

    return y_train_enc, y_test_enc, le

# Train an XGBoost classifier for multiclass (H/D/A) match outcome prediction.
def train_xgboost(X_train: pd.DataFrame,y_train_enc: np.ndarray) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        objective="multi:softprob", # Makes the model output probabilities for each of the three outcomes (H/D/A).
        num_class=3,                # Specifies that there are three possible classes to predict.
        n_estimators=200,           # Number of boosting rounds (trees). More trees allow better learning but increase training time.
        learning_rate=0.05,         # Controls how fast the model learns; a small value makes learning more stable and avoids overfitting.
        max_depth=3,                # Limits how deep each tree can grow, keeping the model simple and reducing overfitting.
        min_child_weight=6,         # Requires a minimum amount of data in each leaf; prevents the model from learning noise.  
        gamma=1,                    # Requires a minimum improvement (gain) to create new splits; reduces unnecessary splits.
        subsample=0.7,              # Trains each tree on a random 70% of the data; helps generalization.
        colsample_bytree=0.7,       # Uses only 70% of the features when building each tree; increases diversity.
        reg_lambda=3.0,             # L2 regularization: makes the model more stable by shrinking large weights.
        reg_alpha=1,                # L1 regularization: can push some weights to zero, simplifying the model.
        random_state=42,            # Ensures results stay the same each time the model is run.
        n_jobs=-1,                  # Uses all CPU cores to speed up training.
        eval_metric="mlogloss",     # The model tries to minimize multiclass log-loss, which is the main performance metric in this project.
    )

    model.fit(X_train, y_train_enc)
    return model

# Compute accuracy, log loss and Brier scores for a fitted model. Also returns predicted probabilities and encoded predictions.
def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test_enc: np.ndarray, le: LabelEncoder) -> dict:
    # step 1: predictions
    y_proba = model.predict_proba(X_test)
    y_pred_enc = model.predict(X_test)

    # step 2: accuracy
    accuracy = accuracy_score(y_test_enc, y_pred_enc)

    # step 3: log loss
    ll = log_loss(y_test_enc, y_proba)

    # step 4: Brier scores per class
    brier_scores = []

    for class_idx, class_label in enumerate(le.classes_):
        y_true_binary = (y_test_enc == class_idx).astype(int)
        y_prob_class = y_proba[:, class_idx]
        brier = brier_score_loss(y_true_binary, y_prob_class)
        brier_scores.append(brier)

    mean_brier = float(np.mean(brier_scores))

    # step 5: Results
    print("  ----- XG Boost Model Evaluation (Test Set) without scaled features -----")
    print(f"     -> Accuracy: {accuracy:.3f}")
    print(f"     -> Log Loss: {ll:.4f}")
    print(f"     -> Mean Brier score: {mean_brier:.4f}")

    return {
        "y_proba": y_proba,
        "y_pred_enc": y_pred_enc,
        "accuracy": float(accuracy),
        "log_loss": float(ll),
        "brier_scores": brier_scores,
        "mean_brier": mean_brier,
    }

# Plot reliability / calibration curves for each class.
def plot_calibration_curves(y_test_enc: np.ndarray, y_proba: np.ndarray, le: LabelEncoder, save_path: Path= CALIBRATION_PLOT_XGB_PATH, n_bins: int = 10,) -> None:
    os.makedirs(save_path.parent, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for class_idx, class_label in enumerate(le.classes_):
        y_true_binary = (y_test_enc == class_idx).astype(int)
        y_prob_class = y_proba[:, class_idx]

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

    plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency (True probability)")
    plt.title("Calibration Curve: XGBoost + Elo")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 
    print(f"  -> Calibration plot saved to: {save_path}")

# Main function to run the full XGBoost modelling pipeline: load, preprocess, train XGBoost, evaluate, and plot calibration.
def run_xgboost_pipeline(features_path: Path = FEATURES_ELO_PATH, plot_path: Path = CALIBRATION_PLOT_XGB_PATH, train_ratio: float = 0.8) -> tuple[xgb.XGBClassifier, dict]:
    print("\n===== Starting XGBoost Model Training (10) =====")

    # --- Confihuration ---
    feature_cols = [
        "is_home",
        "rolling_xg_for_5",
        "rolling_xg_against_5",
        "rolling_xg_diff_5",
        "rolling_points_5",
        "strength_points_5",
        "elo_home_before",
        "elo_away_before",
        "elo_diff_home"
    ]
    
    # 1. Load data
    df = load_long_features_with_elo(features_path)

    # 2. Prepare dataset
    X, y = prepare_dataset(df, feature_cols, target_col="result")

    # 3. Time-based split
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=train_ratio)

    # 4. Preprocessing (Encode target)
    y_train_enc, y_test_enc, le = encode_target(y_train, y_test)

    # 5. Train XGBoost (no scaling needed for XG Boost)
    print("  -> Training XGBoost model...")
    model = train_xgboost(X_train, y_train_enc)
    print("  -> Model trained successfully.")

    # 6. Evaluate
    print("  -> Evaluating model on test set...")
    metrics = evaluate_model(model, X_test, y_test_enc, le)

    # 7. Save calibration curves
    print("  -> Creating and saving calibration curve plot.")
    plot_calibration_curves(
        y_test_enc=y_test_enc,
        y_proba=metrics["y_proba"],
        le=le,
        n_bins=10,
        save_path=plot_path
    )
    print("===== XGBoost Model Training (10) Complete. ✅ =====\n")
    return model, metrics


if __name__ == "__main__":
    run_xgboost_pipeline()