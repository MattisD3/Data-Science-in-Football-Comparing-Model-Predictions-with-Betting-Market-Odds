import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, Dict

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Paths
FEATURES_ELO_PATH = Path("data/processed/features_matches_long_elo_22_23.csv")
DATA_BOOKMAKERS_PATH = Path("data/processed/clean_bookmakers_22_23.csv")
OUT_PATH = Path("results/model_vs_bookmakers_summary_22_23.csv")


# Load long-format match features (two rows per match, one per team).
def load_long_features(path: Path = FEATURES_ELO_PATH ) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded long features: {df.shape}")
    return df

# Drop rows with NaNs in selected features or target, sort by date and return feature matrix X and target vector y.
def prepare_dataset(df: pd.DataFrame, feature_cols: list[str], target_col: str = "result") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

    print("\nShape BEFORE dropna:", df.shape)
    print("NaN per column BEFORE drop:")
    print(df[feature_cols + [target_col]].isna().sum())

    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    print("\nShape AFTER dropna:", df_clean.shape)
    print("NaN per column AFTER drop:")
    print(df_clean[feature_cols + [target_col]].isna().sum())

    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    return X, y, df_clean

# Chronological train/test split (no shuffling) to avoid leakage. The first `train_ratio` fraction of observations are used for training, the remaining ones for testing.
def time_based_split(X: pd.DataFrame, y: pd.Series, df_clean: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    train_size = int(train_ratio * len(X))

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    df_train = df_clean.iloc[:train_size].copy()
    df_test = df_clean.iloc[train_size:].copy()

    print("\nTrain shapes:", X_train.shape, y_train.shape)
    print("Test shapes :", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, df_train, df_test

# Encode match results ('H', 'D', 'A') as integer labels.
def encode_target(y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("\nClasses (LabelEncoder):", le.classes_)  # expected ['A', 'D', 'H']
    print("First 10 encoded y_train:", y_train_enc[:10])

    return y_train_enc, y_test_enc, le

# Standardize features based on TRAIN set only.
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTrain scaled shape:", X_train_scaled.shape)
    print("Test scaled shape :", X_test_scaled.shape)

    print("NaN in X_train_scaled:", np.isnan(X_train_scaled).sum())
    print("NaN in X_test_scaled :", np.isnan(X_test_scaled).sum())

    return X_train_scaled, X_test_scaled, scaler

#  Build the best calibrated model: Random Forest + Platt scaling (sigmoid). We wrap a RandomForestClassifier inside CalibratedClassifierCV
def build_calibrated_random_forest(random_state: int = 42) -> CalibratedClassifierCV:
    base_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=1,
    )

    clf = CalibratedClassifierCV(
        estimator=base_rf,
        method="sigmoid",  # Platt scaling
        cv=3,
    )

    return clf

# Convert team-level probabilities (long format) into match-level probabilities (one row per match, home team only).
def build_match_level_probabilities(df_test: pd.DataFrame, y_proba_test: np.ndarray, le: LabelEncoder) -> pd.DataFrame:
    df_probs = df_test.copy()

    # Map label encoder indices: we expect ['A', 'D', 'H']
    class_to_index: Dict[str, int] = {cls: idx for idx, cls in enumerate(le.classes_)}
    idx_A = class_to_index["A"]
    idx_D = class_to_index["D"]
    idx_H = class_to_index["H"]

    # Attach probabilities for each outcome from home-team perspective
    df_probs["prob_model_home"] = y_proba_test[:, idx_H]
    df_probs["prob_model_draw"] = y_proba_test[:, idx_D]
    df_probs["prob_model_away"] = y_proba_test[:, idx_A]

    # Keep only home rows -> one row per match
    df_home = df_probs[df_probs["is_home"] == 1].copy()
    print("\nTest matches (home rows only):", len(df_home))

    df_home.rename(
        columns={
            "team": "home_team",
            "opponent": "away_team",
        },
        inplace=True,
    )

    # Keep only relevant columns
    cols_keep = [
        "date",
        "home_team",
        "away_team",
        "result",
        "prob_model_home",
        "prob_model_draw",
        "prob_model_away",
    ]
    df_home = df_home[cols_keep]

    return df_home

# Load bookmaker odds and implied probabilities (wide format, one row per match).
def load_bookmakers(path: Path = DATA_BOOKMAKERS_PATH) -> pd.DataFrame:
    df_book = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded bookmakers data: {df_book.shape}")
    return df_book

# Merge bookmaker probabilities with model probabilities on (date, home_team, away_team).
def merge_model_and_bookmakers(df_book: pd.DataFrame, df_match_probs: pd.DataFrame) -> pd.DataFrame:
    merge_cols = ["date", "home_team", "away_team"]

    df_merge = df_book.merge(
        df_match_probs,
        on=merge_cols,
        how="inner",
        validate="one_to_one",
    )

    print("\nMerged shape (bookmakers + model on test set):", df_merge.shape)

    return df_merge

# Compute RMSE and MAE between model and bookmaker probabilities for each outcome (H, D, A).
def compute_rmse_mae(df_merge: pd.DataFrame) -> pd.DataFrame:
    metrics = []

    pairs = [
        ("H", "prob_model_home", "prob_book_home"),
        ("D", "prob_model_draw", "prob_book_draw"),
        ("A", "prob_model_away", "prob_book_away"),
    ]

    for outcome, col_model, col_book in pairs:
        errors = df_merge[col_model] - df_merge[col_book]
        rmse = np.sqrt(mean_squared_error(df_merge[col_book], df_merge[col_model]))
        mae = mean_absolute_error(df_merge[col_book], df_merge[col_model])

        metrics.append(
            {
                "outcome": outcome,
                "RMSE_model_vs_book": float(rmse),
                "MAE_model_vs_book": float(mae),
            }
        )

    df_metrics = pd.DataFrame(metrics)
    print("\nRMSE / MAE summary (model vs bookmakers):")
    print(df_metrics)

    return df_metrics

# Scatter plots: model probability vs bookmaker probability (H/D/A). Diagonal line means perfect agreement.
def plot_model_vs_book_scatter(df_merge: pd.DataFrame, save_path=None) -> None:
    outcomes = [
        ("H", "prob_model_home", "prob_book_home"),
        ("D", "prob_model_draw", "prob_book_draw"),
        ("A", "prob_model_away", "prob_book_away"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)

    for ax, (label, col_model, col_book) in zip(axes, outcomes):
        ax.scatter(df_merge[col_book], df_merge[col_model], alpha=0.7)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("Bookmakers probability")
        ax.set_ylabel("Model probability")
        ax.set_title(f"{label} - model vs bookmakers")

    fig.suptitle("Model vs Bookmakers probabilities (test set)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close(fig)
    
# Histograms of errors (model_prob - book_prob) for H/D/A. Centered around 0 means no systematic bias.
def plot_error_distributions(df_merge: pd.DataFrame, save_path=None) -> None:
    outcomes = [
        ("H", "prob_model_home", "prob_book_home"),
        ("D", "prob_model_draw", "prob_book_draw"),
        ("A", "prob_model_away", "prob_book_away"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (label, col_model, col_book) in zip(axes, outcomes):
        errors = df_merge[col_model] - df_merge[col_book]
        ax.hist(errors, bins=15, alpha=0.8)
        ax.axvline(0, color="k", linestyle="--")
        ax.set_title(f"Error distribution - {label}")
        ax.set_xlabel("Model prob - Book prob")
        ax.set_ylabel("Number of matches")

    fig.suptitle("Error distributions: model vs bookmakers (H/D/A)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close(fig)

# Reliability / calibration curves for model and bookmakers for each outcome (H/D/A).
def plot_calibration_model_vs_book(df_merge: pd.DataFrame, save_path=None) -> None:
    outcomes = [
        ("H", "prob_model_home", "prob_book_home"),
        ("D", "prob_model_draw", "prob_book_draw"),
        ("A", "prob_model_away", "prob_book_away"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (label, col_model, col_book) in zip(axes, outcomes):
        # True binary labels for this outcome
        y_true = (df_merge["result"] == label).astype(int)

        # Model calibration
        true_m, pred_m = calibration_curve(
            y_true, df_merge[col_model], n_bins=8, strategy="uniform"
        )

        # Bookmakers calibration
        true_b, pred_b = calibration_curve(
            y_true, df_merge[col_book], n_bins=8, strategy="uniform"
        )

        ax.plot(pred_m, true_m, marker="o", label="Model")
        ax.plot(pred_b, true_b, marker="o", label="Bookmakers")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")

        ax.set_title(f"Calibration - {label}")
        ax.set_xlabel(f"Predicted probability ({label})")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Reliability curves: model vs bookmakers (test set)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close(fig)
    
# Overlayed histograms of model vs bookmaker probabilities for H/D/A.
def plot_probability_distributions(df_merge: pd.DataFrame, save_path=None) -> None:
    outcomes = [
        ("H", "prob_model_home", "prob_book_home"),
        ("D", "prob_model_draw", "prob_book_draw"),
        ("A", "prob_model_away", "prob_book_away"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, (label, col_model, col_book) in zip(axes, outcomes):
        ax.hist(
            df_merge[col_model],
            bins=15,
            alpha=0.7,
            label="Model",
        )
        ax.hist(
            df_merge[col_book],
            bins=15,
            alpha=0.7,
            label="Bookmakers",
        )
        ax.set_title(f"Probabilities distribution - {label}")
        ax.set_xlabel(f"P({label})")
        ax.set_ylabel("Number of matches")

    fig.suptitle("Distribution of probabilities: model vs bookmakers (H/D/A)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close(fig)

# Full end-to-end comparison between calibrated RF model and bookmakers.
def run_comparison(random_state: int = 42) -> None:
    # 1) Load long-format features
    df_long = load_long_features()

    # 2) Features used by the calibrated model (same as in previous src)
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

    # 3) Prepare dataset
    X, y, df_clean = prepare_dataset(df_long, feature_cols, target_col="result")

    # 4) Time-based split
    X_train, X_test, y_train, y_test, df_train, df_test = time_based_split(X, y, df_clean, train_ratio=0.8)

    # 5) Encode target
    y_train_enc, y_test_enc, le = encode_target(y_train, y_test)

    # 6) Scale features
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 7) Build and train calibrated Random Forest
    clf = build_calibrated_random_forest(random_state=random_state)
    clf.fit(X_train_scaled, y_train_enc)
    print("\nCalibrated Random Forest (Platt) trained!")

    # 8) Predicted probabilities on test set
    y_proba_test = clf.predict_proba(X_test_scaled)
    y_pred_test = clf.predict(X_test_scaled)

    print("\nPredicted probabilities shape:", y_proba_test.shape)
    print("Encoded predictions sample:", y_pred_test[:10])
    print("True encoded labels sample:", y_test_enc[:10])

    # 9) Build match-level probabilities (home rows only)
    df_match_probs = build_match_level_probabilities(df_test, y_proba_test, le)

    # 10) Load bookmakers and merge
    df_book = load_bookmakers()
    df_merge = merge_model_and_bookmakers(df_book, df_match_probs)

    # 11) Compute RMSE / MAE summary
    df_metrics = compute_rmse_mae(df_merge)

    # 12) Plots: scatter, errors, calibration, distributions
    results_dir=Path("results")
    scatter_path = results_dir / "scatter_model_vs_book_22_23.png"
    errors_path = results_dir / "errors_model_vs_book_22_23.png"
    calib_path = results_dir / "calibration_model_vs_book_22_23.png"
    dist_path = results_dir / "prob_dist_model_vs_book_22_23.png"

    plot_model_vs_book_scatter(df_merge, save_path=scatter_path)
    plot_error_distributions(df_merge, save_path=errors_path)
    plot_calibration_model_vs_book(df_merge, save_path=calib_path)
    plot_probability_distributions(df_merge, save_path=dist_path)
    

    # 13) Save metrics summary
    df_metrics.to_csv(OUT_PATH, index=False)
    print(f"\nSummary metrics saved to: {OUT_PATH}")
    print(f"Plots saved to: {results_dir}")



if __name__ == "__main__":
    run_comparison()
