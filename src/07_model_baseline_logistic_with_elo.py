import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Path
FEATURES_PATH = "data/processed/features_matches_long_22_23.csv"
ELO_PATH = "data/processed/elo_rating_22_23.csv"

# Load engineered long-format match features.
def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Initial shape (long features): {df.shape}")
    return df

# Load match-level Elo ratings (one row per match).
def load_elo(path: str = ELO_PATH) -> pd.DataFrame:
    df_elo = pd.read_csv(path, parse_dates=["date"])
    print(f"Elo file shape: {df_elo.shape}")
    return df_elo

# Merge match-level Elo into team-level long-format dataset.
def add_elo_features_to_long(df_long: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
    """
    For each row (team perspective) we create:
    - elo_team_before: Elo of this team before the match
    - elo_opponent_before: Elo of the opponent before the match
    - elo_diff_before: team Elo - opponent Elo
    """
    # Merge on match_id to bring home/away Elo
    df_merged = df_long.merge(
        df_elo[[
            "match_id",
            "home_team",
            "away_team",
            "elo_home_before",
            "elo_away_before",
            "elo_diff_home",
        ]],
        on="match_id",
        how="left",
    )

    # Sanity check
    if df_merged["elo_home_before"].isna().any():
        print("Warning: some rows have missing Elo after merge.")

    # Team Elo before the match (depends on is_home)
    df_merged["elo_team_before"] = np.where(
        df_merged["is_home"] == 1,
        df_merged["elo_home_before"],
        df_merged["elo_away_before"],
    )

    # Opponent Elo before the match
    df_merged["elo_opponent_before"] = np.where(
        df_merged["is_home"] == 1,
        df_merged["elo_away_before"],
        df_merged["elo_home_before"],
    )

    # Elo difference from the team's point of view
    df_merged["elo_diff_before"] = df_merged["elo_team_before"] - df_merged["elo_opponent_before"]

    print("Shape after adding Elo to long:", df_merged.shape)
    return df_merged

# Select features/target, handle NaNs, sort by date.
def prepare_dataset(df: pd.DataFrame, feature_cols: list[str], target_col: str = "result") -> tuple[pd.DataFrame, pd.Series]:
    # Drop rows with NaNs in features or target
    print("\nShape BEFORE dropna:", df.shape)
    print("NaN per column BEFORE drop:")
    print(df[feature_cols + [target_col]].isna().sum())

    df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    print("\nShape AFTER dropna:", df.shape)
    print("NaN per column AFTER drop:")
    print(df[feature_cols + [target_col]].isna().sum())

    # Ensure date is datetime and sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    return X, y

# Split train/test by time (no shuffling, otherways risk of leakage).
def time_based_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_size = int(train_ratio * len(X))

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    print("\nTrain shapes:", X_train.shape, y_train.shape)
    print("Test shapes :", X_test.shape, y_test.shape)

    # Sanity check: no NaNs
    print("\nNaN in TRAIN features:")
    print(X_train.isna().sum())

    print("\nNaN in TEST features:")
    print(X_test.isna().sum())

    return X_train, X_test, y_train, y_test

# Encode match results (H/D/A) as integers.
def encode_target(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("\nClasses:", le.classes_)  # expected ['A', 'D', 'H']
    print("First 10 encoded y_train:", y_train_enc[:10])

    return y_train_enc, y_test_enc, le

# Standardize features based on TRAIN set only.
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTrain scaled shape:", X_train_scaled.shape)
    print("Test scaled shape :", X_test_scaled.shape)

    # Check NaNs after scaling
    print("NaN in X_train_scaled:", np.isnan(X_train_scaled).sum())
    print("NaN in X_test_scaled :", np.isnan(X_test_scaled).sum())

    return X_train_scaled, X_test_scaled, scaler

# Train multinomial logistic regression.
def train_logistic_model(X_train_scaled: np.ndarray, y_train_enc: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500,
    )
    model.fit(X_train_scaled, y_train_enc)
    print("\nModel trained (with Elo features)!")
    return model

# Compute accuracy, log loss, Brier scores and return everything (including probabilities) in a dict.
def evaluate_model(model: LogisticRegression, X_test_scaled: np.ndarray, y_test_enc: np.ndarray, le: LabelEncoder) -> dict:

    # Step 1: predictions
    y_proba = model.predict_proba(X_test_scaled)
    y_pred_enc = model.predict(X_test_scaled)

    print("\nPredicted probabilities shape:", y_proba.shape)
    print("Encoded predictions sample:", y_pred_enc[:10])
    print("True encoded labels sample:", y_test_enc[:10])

    # Step 2: accuracy
    accuracy = accuracy_score(y_test_enc, y_pred_enc)
    print("\nAccuracy on test set (with Elo):", round(accuracy, 3))

    # Step 3: log loss
    ll = log_loss(y_test_enc, y_proba)
    print("Log loss on test set (with Elo):", round(ll, 4))

    # Step 4: Brier scores per class
    brier_scores = []
    for class_idx, class_label in enumerate(le.classes_):
        y_true_binary = (y_test_enc == class_idx).astype(int)
        y_prob_class = y_proba[:, class_idx]
        brier = brier_score_loss(y_true_binary, y_prob_class)
        brier_scores.append(brier)
        print(f"Brier score for class {class_label} (with Elo): {brier:.4f}")

    mean_brier = float(np.mean(brier_scores))
    print("Mean Brier score (with Elo):", round(mean_brier, 4))

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
            label=f"Class {class_label}",
        )

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "--", color="black", label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration curves (multinomial logistic regression + Elo)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the full baseline pipeline with Elo: load, train, evaluate, plot.
def run_baseline_with_elo():
    # Load data
    df_long = load_features(FEATURES_PATH)
    df_elo = load_elo(ELO_PATH)

    # Add Elo features to the long-format dataset
    df_with_elo = add_elo_features_to_long(df_long, df_elo)

    # Base features (same as previous baseline)
    base_feature_cols = [
        "is_home",
        "rolling_xg_for_5",
        "rolling_xg_against_5",
        "rolling_xg_diff_5",
        "rolling_points_5",
        "strength_points_5",
    ]

    # New Elo features
    elo_feature_cols = [
        "elo_team_before",
        "elo_opponent_before",
        "elo_diff_before",
    ]

    feature_cols = base_feature_cols + elo_feature_cols

    # Prepare dataset
    X, y = prepare_dataset(df_with_elo, feature_cols, target_col="result")

    # Time-based split
    X_train, X_test, y_train, y_test = time_based_split(X, y, train_ratio=0.8)

    # Encode target
    y_train_enc, y_test_enc, le = encode_target(y_train, y_test)

    # Scale features
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # Train model
    model = train_logistic_model(X_train_scaled, y_train_enc)

    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test_enc, le)

    # Plot calibration curves
    plot_calibration_curves(
        y_test_enc=y_test_enc,
        y_proba=metrics["y_proba"],
        le=le,
        n_bins=10,
    )

    return metrics


if __name__ == "__main__":
    run_baseline_with_elo()

