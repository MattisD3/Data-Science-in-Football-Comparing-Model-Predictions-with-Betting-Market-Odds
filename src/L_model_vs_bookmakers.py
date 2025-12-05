import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Optional

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Input path
FEATURES_ELO_PATH = Path("data/processed/features_matches_long_elo_22_23.csv")
DATA_BOOKMAKERS_PATH = Path("data/processed/clean_bookmakers_22_23.csv")
# Output path
SUMMARY_RMSE_MAE_PATH = Path("results/model_vs_bookmakers_summary_22_23.csv")
# Output: plots — model vs bookmakers
SCATTER_PLOT_PATH = Path("results/scatter_model_vs_book_22_23.png")
ERRORS_PLOT_PATH = Path("results/errors_model_vs_book_22_23.png")
CALIB_PLOT_PATH = Path("results/calibration_model_vs_book_22_23.png")
PROB_DIST_PLOT_PATH = Path("results/prob_dist_model_vs_book_22_23.png")

# Output: team-level bias plots
TEAM_HOME_BIAS_BAR_PATH = Path("results/team_home_bias_bar_22_23.png")
TEAM_HOME_BIAS_HEATMAP_PATH = Path("results/team_home_bias_heatmap_22_23.png")
TEAM_AWAY_BIAS_BAR_PATH = Path("results/team_away_bias_bar_22_23.png")
TEAM_AWAY_BIAS_HEATMAP_PATH = Path("results/team_away_bias_heatmap_22_23.png")
GLOBAL_BIAS_BAR_PATH = Path("results/global_bias_bar_22_23.png")

# Output: Big Six vs Others bias
BIG_SIX_HOME_PATH = Path("results/big_six_vs_others_home_bias_22_23.png")
BIG_SIX_AWAY_PATH = Path("results/big_six_vs_others_away_bias_22_23.png")


# Load long-format match features (two rows per match, one per team).
def load_long_features(path: Path = FEATURES_ELO_PATH ) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  -> Loaded long features: {df.shape}")
    return df

# Drop rows with NaNs in selected features or target, sort by date and return feature matrix X and target vector y.
def prepare_dataset(df: pd.DataFrame, feature_cols: list[str], target_col: str = "result") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df_clean = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)

    # Ensure date is datetime and sort chronologically
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

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

    print(f"  -> Train size: {len(X_train)} | Test size: {len(X_test)} ")

    return X_train, X_test, y_train, y_test, df_train, df_test

# Encode match results ('H', 'D', 'A') as integer labels.
def encode_target(y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print(f"  -> Target encoded. Classes: {le.classes_}")

    return y_train_enc, y_test_enc, le

# Standardize features based on TRAIN set only.
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  -> Features scaled. Scaled Train shape: {X_train_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler

# Build the best calibrated model: Multinomial logistic regression + Platt scaling (sigmoid). We wrap a LogisticRegression estimator inside CalibratedClassifierCV.
def build_calibrated_logistic(random_state: int = 42) -> CalibratedClassifierCV:
    base_log = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        random_state=random_state,
        n_jobs=-1,          
    )
    # Wrap with Platt scaling (sigmoid) using CV=3
    clf = CalibratedClassifierCV(
        estimator=base_log,
        method="sigmoid",   # Platt scaling
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

    df_home.rename(columns={"team": "home_team", "opponent": "away_team"}, inplace=True)

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
    print(f"  -> Build match-level probabilities (one row per match): {len(df_home)} matches.")
    return df_home

# Load bookmaker odds and implied probabilities (wide format, one row per match).
def load_bookmakers(path: Path = DATA_BOOKMAKERS_PATH) -> pd.DataFrame:
    df_book = pd.read_csv(path, parse_dates=["date"])
    print(f"  -> Loaded bookmakers data : {df_book.shape}")
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

    
    print(f"  -> Merged shape (bookmakers + model on test set): {df_merge.shape[0]} matches")

    return df_merge

# Compute RMSE and MAE between my model and bookmaker probabilities for each outcome (H, D, A).
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
        ax.grid(True, alpha=0.5)

    fig.suptitle("Model vs Bookmakers probabilities (test set)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close(fig)
        print(f"  -> Scatter plot saved to : {save_path}")

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
        ax.grid(True, alpha=0.5)

    fig.suptitle("Error distributions: model vs bookmakers (H/D/A)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close(fig)
        print(f"  -> Error distribution plot saved to {save_path}")

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
        true_m, pred_m = calibration_curve(y_true, df_merge[col_model], n_bins=8, strategy="uniform")
        
        # Bookmakers calibration
        true_b, pred_b = calibration_curve(y_true, df_merge[col_book], n_bins=8, strategy="uniform")

        ax.plot(pred_m, true_m, marker="o", label="Model")
        ax.plot(pred_b, true_b, marker="o", label="Bookmakers")
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")

        ax.set_title(f"Calibration - {label}")
        ax.set_xlabel(f"Predicted probability ({label})")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.5)

    fig.suptitle("Reliability curves: model vs bookmakers (test set)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close(fig)
        print(f"  -> Calibration comparison plot saved to: {save_path}")

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
        ax.grid(True, alpha=0.5)

    fig.suptitle("Distribution of probabilities: model vs bookmakers (H/D/A)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,dpi=300)
        plt.close(fig)
        print(f"  -> Probability distribution plot saved to: {save_path}")

# Add per-match error columns (model - bookmakers) for H/D/A.
def add_error_columns(df_merge: pd.DataFrame) -> pd.DataFrame:
    df = df_merge.copy()
    df["err_home"] = df["prob_model_home"] - df["prob_book_home"]
    df["err_draw"] = df["prob_model_draw"] - df["prob_book_draw"]
    df["err_away"] = df["prob_model_away"] - df["prob_book_away"]
    print("  -> Error columns (model - bookmakers) added.")
    return df

# Compute average prediction errors for each HOME team. This shows whether the model systematically OVERestimates or UNDERestimates teams when they play at home.
def compute_team_home_bias(df_errors: pd.DataFrame) -> pd.DataFrame:
    team_bias = (df_errors.groupby("home_team")[["err_home", "err_draw", "err_away"]].mean())
    return team_bias

# Plot bar chart of HOME-team bias (sorted). Displays how each team is over/underestimated when playing at home.
def plot_team_home_bias_bar(team_bias: pd.DataFrame, save_path=None) -> None:
    team_bias_sorted = team_bias.sort_values("err_home", ascending=True)
    ax = team_bias_sorted["err_home"].plot(kind="bar", figsize=(14, 5))
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Model vs Bookmakers bias - Home Win probability (home perspective)")
    plt.ylabel("Mean error (model - bookmakers)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> Home-team bias bar plot saved to: {save_path}")

# Heatmap of HOME-team bias (err_home, err_draw, err_away). Sorted so that the most underestimated home teams (most negative bias) appear at the top.
def plot_team_bias_heatmap(team_bias: pd.DataFrame, save_path=None) -> None:
    team_bias_sorted = team_bias.sort_values("err_home", ascending=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        team_bias_sorted,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".3f",
    )
    plt.title("Team-level model vs bookmakers divergence (mean error) - Home perspective")
    plt.ylabel("home_team")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> Home-team bias heatmap saved to: {save_path}")

# Compute average prediction errors for each HOME team. This shows whether the model systematically OVERestimates or UNDERestimates teams when they play away.
def compute_team_away_bias(df_errors: pd.DataFrame) -> pd.DataFrame:
    team_bias_away = (df_errors.groupby("away_team")[["err_home", "err_draw", "err_away"]].mean().rename_axis("away_team"))
    return team_bias_away

# Plot bar chart of AWAY-team bias (sorted). Displays how each team is over/underestimated when playing away.
def plot_team_away_bias_bar(team_bias_away: pd.DataFrame, save_path=None) -> None:
    team_bias_sorted = team_bias_away.sort_values("err_away", ascending=True)

    plt.figure(figsize=(14, 5))
    team_bias_sorted["err_away"].plot(kind="bar")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Model vs Bookmakers bias — Away Win probability (away perspective)")
    plt.ylabel("Mean error (model - bookmakers)")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> Away-team bias bar plot saved to: {save_path}")

# Heatmap of AWAY-team bias (err_home / err_draw / err_away) from the p.o.v. of the AWAY team. Sorted so that the most underestimated away teams (most negative bias) appear at the top.
def plot_team_away_bias_heatmap(team_bias_away: pd.DataFrame, save_path=None) -> None:
    team_bias_sorted = team_bias_away.sort_values("err_away", ascending=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        team_bias_sorted,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".3f",
    )
    plt.title("Team-level model vs bookmakers divergence (mean error) — Away perspective")
    plt.ylabel("away_team")
    plt.tight_layout()

    if save_path :
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> Away-team bias heatmap saved to: {save_path}")

# Compute global home / draw / away bias.
def compute_global_bias(df_errors: pd.DataFrame) -> Tuple[float, float, float]:
    home_bias = float(df_errors["err_home"].mean())
    draw_bias = float(df_errors["err_draw"].mean())
    away_bias = float(df_errors["err_away"].mean())
    return home_bias, draw_bias, away_bias

# Plot global bias bar chart.
def plot_global_bias_bar(home_bias: float, draw_bias: float, away_bias: float, save_path=None) -> None:
    biases = [home_bias, draw_bias, away_bias]
    labels = ["Home", "Draw", "Away"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, biases)
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Systematic model vs bookmakers bias")
    plt.ylabel("Mean error (model - bookmakers)")
    plt.tight_layout()
    if save_path :
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> Global bias bar plot saved to: {save_path}")

# Compute Big Six vs others bias (home and away perspective).
def compute_big_six_bias(df_errors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    big_six = {
        "Arsenal",
        "Chelsea",
        "Liverpool",
        "Manchester City",
        "Manchester United",
        "Tottenham Hotspur",
    }

    # Home perspective: group by whether home_team is Big Six.
    df_errors["is_big_six_home"] = df_errors["home_team"].isin(big_six)

    home_bias_group = (
        df_errors.groupby(
            np.where(df_errors["is_big_six_home"], "Big Six (home)", "Other teams")
        )[["err_home", "err_draw", "err_away"]]
        .mean()
        .rename_axis("group")
    )

    # Away perspective: group by whether AWAY team is Big Six.
    df_errors["is_big_six_away"] = df_errors["away_team"].isin(big_six)

    away_bias_group = (
        df_errors.groupby(
            np.where(df_errors["is_big_six_away"], "Big Six (away)", "Other teams")
        )[["err_home", "err_draw", "err_away"]]
        .mean()
        .rename_axis("group")
    )

    return home_bias_group, away_bias_group

# Plot Big Six vs others bias (stacked outcome bars).
def plot_big_six_bias(bias_home: pd.DataFrame, bias_away: pd.DataFrame, save_path_home=None, save_path_away=None) -> None:
    # Home team perspective
    plt.figure(figsize=(8, 5))
    bias_home[["err_home", "err_draw", "err_away"]].plot(kind="bar", figsize=(8, 5))
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Model vs Bookmakers bias — Big Six vs Others (Home team)")
    plt.ylabel("Mean error (model - bookmakers)")
    plt.tight_layout()
    if save_path_home:
        plt.savefig(save_path_home, dpi=300)
        plt.close()
        print(f"  -> Big Six vs Others (home) bias plot saved to: {save_path_home}")

    # Away team perspective
    plt.figure(figsize=(8, 5))
    bias_away[["err_home", "err_draw", "err_away"]].plot(kind="bar", figsize=(8, 5))
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Model vs Bookmakers bias — Big Six vs Others (Away team)")
    plt.ylabel("Mean error (model - bookmakers)")
    plt.tight_layout()
    if save_path_away:
        plt.savefig(save_path_away, dpi=300)
        plt.close()
        print(f"  -> Big Six vs Others (away) bias plot saved to: {save_path_away}")
    

def run_comparison(random_state: int = 42) -> None:
    print("\n===== Starting the comparison between my best-calibrated model (logistic regression + Platt) and the bookmakers' probabilities. (12) =====")

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
    # 1. Load long-format features
    df_long = load_long_features()

    # 2. Prepare dataset
    X, y, df_clean = prepare_dataset(df_long, feature_cols, target_col="result")

    # 3. Time-based split
    X_train, X_test, y_train, y_test, df_train, df_test = time_based_split(X, y, df_clean, train_ratio=0.8)

    # 4. Preprocessing (Encode target + scale features)
    y_train_enc, y_test_enc, le = encode_target(y_train, y_test)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 5. Train calibrated Multionomial Logistic Regression
    print("  -> Training Calibrated (Platt) Multionomial Logistic Regression model ...")
    clf = build_calibrated_logistic(random_state=random_state)
    clf.fit(X_train_scaled, y_train_enc)
    print("  -> Model trained successfully.")

    # 6. Predicted probabilities on test set
    y_proba_test = clf.predict_proba(X_test_scaled)
    y_pred_test = clf.predict(X_test_scaled)

    # 7. Build match-level probabilities (home rows only)
    df_match_probs = build_match_level_probabilities(df_test, y_proba_test, le)

    # 8. Load bookmakers and merge
    df_book = load_bookmakers()
    df_merge = merge_model_and_bookmakers(df_book, df_match_probs)

    # 9. Compute RMSE / MAE summary
    
    df_metrics = compute_rmse_mae(df_merge)
    print("\n  ----- RMSE/MAE summary (Model vs Bookmakers) -----")
    print(df_metrics.to_string(index=False))

    # 10. Plots: scatter, errors, calibration, distributions
    plot_model_vs_book_scatter(df_merge, SCATTER_PLOT_PATH)
    plot_error_distributions(df_merge, ERRORS_PLOT_PATH)
    plot_calibration_model_vs_book(df_merge, CALIB_PLOT_PATH)
    plot_probability_distributions(df_merge, PROB_DIST_PLOT_PATH)

    # 11. Team-level & Big Six divergence analysis
    df_errors = add_error_columns(df_merge)

    # a) Team bias – HOME team perspective
    team_home_bias = compute_team_home_bias(df_errors)
    plot_team_home_bias_bar(team_home_bias, TEAM_HOME_BIAS_BAR_PATH)
    plot_team_bias_heatmap(team_home_bias, TEAM_HOME_BIAS_HEATMAP_PATH)

    # b) Team bias – AWAY team perspective
    team_away_bias = compute_team_away_bias(df_errors)
    plot_team_away_bias_bar(team_away_bias, TEAM_AWAY_BIAS_BAR_PATH)
    plot_team_away_bias_heatmap(team_away_bias, TEAM_AWAY_BIAS_HEATMAP_PATH)

    # c) Global home/draw/away bias
    home_bias, draw_bias, away_bias = compute_global_bias(df_errors)
    print("\n  ----- Global bias (mean model - bookmakers error) -----")
    print(f"     -> Home bias: {home_bias:.4f}")
    print(f"     -> Draw bias: {draw_bias:.4f}")
    print(f"     -> Away bias: {away_bias:.4f}")

    plot_global_bias_bar(home_bias, draw_bias, away_bias, GLOBAL_BIAS_BAR_PATH)

    # d) Big Six vs Other teams
    bias_home, bias_away = compute_big_six_bias(df_errors)
    plot_big_six_bias(bias_home, bias_away, BIG_SIX_HOME_PATH, BIG_SIX_AWAY_PATH)

    # 12. Save metrics summary (RMSE / MAE)
    df_metrics.to_csv(SUMMARY_RMSE_MAE_PATH, index=False)
    print(f"\nSummary metrics saved to: {SUMMARY_RMSE_MAE_PATH}")
    print("===== Comparison Best Model vs Bookmakers (12) Complete. ✅ =====\n")


if __name__ == "__main__":
    run_comparison()
