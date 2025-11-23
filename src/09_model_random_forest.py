import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Path
FEATURES_ELO_PATH = "data/processed/features_matches_long_elo_22_23.csv"


# Load long-format match features already enriched with Elo (two rows per match, one per team).
def load_long_features_with_elo(path: str = FEATURES_ELO_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded long features: {df.shape}")
    return df

# Drop rows with NaNs in selected features or target, sort by date and return feature matrix X and target vector y.
def prepare_dataset(df: pd.DataFrame, feature_cols: list[str], target_col: str = "result") -> tuple[pd.DataFrame, pd.Series]:
    
    print("\nShape BEFORE dropna:", df.shape)
    print("NaN per column BEFORE drop:")
    print(df[feature_cols + [target_col]].isna().sum())

    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    print("\nShape AFTER dropna:", df_clean.shape)
    print("NaN per column AFTER drop:")
    print(df_clean[feature_cols + [target_col]].isna().sum())

    df_clean["date"] = pd.to_datetime(df_clean["date"])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    return X, y

# Chronological train/test split (no shuffling) to avoid leakage. The first `train_ratio` fraction of matches are used for training,the remaining for testing.
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

# Train a RandomForest classifier for multiclass (H/D/A) match outcome prediction.
def train_random_forest(X_train: pd.DataFrame, y_train_enc: np.ndarray) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,  # use all available CPU cores
    )

    rf.fit(X_train, y_train_enc)
    print("\nRandom Forest model trained!")
    return rf

# Compute accuracy, log loss and Brier scores for a fitted model. Also returns predicted probabilities and encoded predictions.
def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test_enc: np.ndarray, le: LabelEncoder) -> dict:
    # Predicted probabilities and labels
    y_proba = model.predict_proba(X_test)
    y_pred_enc = model.predict(X_test)

    print("\nPredicted probabilities shape:", y_proba.shape)
    print("Encoded predictions sample:", y_pred_enc[:10])
    print("True encoded labels sample:", y_test_enc[:10])

    # Accuracy
    accuracy = accuracy_score(y_test_enc, y_pred_enc)
    print("\nAccuracy on test set (Random Forest):", round(accuracy, 3))

    # Log loss
    ll = log_loss(y_test_enc, y_proba)
    print("Log loss on test set (Random Forest):", round(ll, 4))

    # Brier scores per class
    brier_scores = []
    for class_idx, class_label in enumerate(le.classes_):
        y_true_binary = (y_test_enc == class_idx).astype(int)
        y_prob_class = y_proba[:, class_idx]
        brier = brier_score_loss(y_true_binary, y_prob_class)
        brier_scores.append(brier)
        print(f"Brier score for class {class_label} (RF): {brier:.4f}")

    mean_brier = float(np.mean(brier_scores))
    print("Mean Brier score (RF):", round(mean_brier, 4))

    return {
        "y_proba": y_proba,
        "y_pred_enc": y_pred_enc,
        "accuracy": float(accuracy),
        "log_loss": float(ll),
        "brier_scores": brier_scores,
        "mean_brier": mean_brier,
    }

# Plot reliability / calibration curves for each class.
def plot_calibration_curves(y_test_enc: np.ndarray, y_proba: np.ndarray, le: LabelEncoder, n_bins: int = 10) -> None:
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
            label=f"Class {class_label}",
            )

    plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curves (Random Forest)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Full Random Forest pipeline:
def run_random_forest_pipeline():

    # Load data
    df = load_long_features_with_elo(FEATURES_ELO_PATH)
    

    # Feature set: rolling stats + Elo + is_home
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

    # Prepare dataset
    X, y = prepare_dataset(df, feature_cols, target_col="result")

    # Time-based split
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=0.8)

    # Encode target
    y_train_enc, y_test_enc, le = encode_target(y_train, y_test)

    # Train Random Forest (no scaling needed)
    model = train_random_forest(X_train, y_train_enc)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test_enc, le)

    # Calibration curves
    plot_calibration_curves(
        y_test_enc=y_test_enc,
        y_proba=metrics["y_proba"],
        le=le,
        n_bins=10,
    )

    return metrics

if __name__ == "__main__":
    run_random_forest_pipeline()
