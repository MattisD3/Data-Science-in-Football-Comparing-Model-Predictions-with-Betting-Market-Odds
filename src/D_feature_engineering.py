import pandas as pd
import numpy as np
from pathlib import Path

# Input path
MATCHES_LONG_PATH = "data/processed/matches_long_22_23.csv"
# Output path
FEATURES_OUTPUT_PATH = "data/processed/features_matches_long_22_23.csv"

# Load long-format match data (two rows per match: one per team).
def load_long_matches(path: str = MATCHES_LONG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values(["team", "date"]).reset_index(drop=True) # Make sure data is sorted for rolling calculations
    print(f"  -> Loaded long matches: {df.shape}")
    return df

# Add basic difference columns like goal_diff and xg_diff.
def add_basic_diff_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["goal_diff"] = df["goals_for"] - df["goals_against"]
    df["xg_diff"] = df["xg_for"] - df["xg_against"]
    return df

# Add points column (3/1/0) from the team perspective.
def add_points_column(df: pd.DataFrame) -> pd.DataFrame:
    def get_points(row):
        # Home team's perspective
        if row["is_home"] == 1:
            if row["result"] == "H": return 3 # Home win (H)
            if row["result"] == "D": return 1 # Draw (D)
            return 0 # Loss (A)
        
        # Away team's perspective
        if row["is_home"] == 0:
            if row["result"] == "A": return 3 # Away win (A)
            if row["result"] == "D": return 1 # Draw (D)
            return 0 # Loss (H)
        
        return 0

    df = df.copy()
    df["points"] = df.apply(get_points, axis=1)
    return df

# Add rolling xG features over the last N games per team (no leakage).
def add_rolling_xg_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()

    df["rolling_xg_for_5"] = np.nan
    df["rolling_xg_against_5"] = np.nan
    df["rolling_xg_diff_5"] = np.nan

    for team in df["team"].unique():
        team_mask = df["team"] == team

        # xG for
        team_xg_for = df.loc[team_mask, "xg_for"]
        df.loc[team_mask, "rolling_xg_for_5"] = (team_xg_for.shift(1).rolling(window).mean())

        # xG against
        team_xg_against = df.loc[team_mask, "xg_against"]
        df.loc[team_mask, "rolling_xg_against_5"] = (team_xg_against.shift(1).rolling(window).mean())

        # xG diff
        team_xg_diff = df.loc[team_mask, "xg_diff"]
        df.loc[team_mask, "rolling_xg_diff_5"] = (team_xg_diff.shift(1).rolling(window).mean())

    return df

# Add rolling points (total, home, away, strength) over last N games per team.
def add_rolling_points_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()

    # Total rolling points (all games)
    df["rolling_points_5"] = np.nan
    for team in df["team"].unique():
        team_mask = df["team"] == team
        team_points = df.loc[team_mask, "points"]

        df.loc[team_mask, "rolling_points_5"] = (team_points.shift(1).rolling(window).sum())

    # Rolling home points (only home games)
    df["rolling_home_points_5"] = np.nan
    for team in df["team"].unique():
        team_home_mask = (df["team"] == team) & (df["is_home"] == 1)
        team_home_points = df.loc[team_home_mask, "points"]

        df.loc[team_home_mask, "rolling_home_points_5"] = (team_home_points.shift(1).rolling(window).sum())

    # Rolling away points (only away games)
    df["rolling_away_points_5"] = np.nan
    for team in df["team"].unique():
        team_away_mask = (df["team"] == team) & (df["is_home"] == 0)
        team_away_points = df.loc[team_away_mask, "points"]

        df.loc[team_away_mask, "rolling_away_points_5"] = ( team_away_points.shift(1).rolling(window).sum())
    
    # Compute contextual strength: home strength if team plays at home, away strength otherwise
    df["strength_points_5"] = np.where(
        df["is_home"] == 1,
        df["rolling_home_points_5"],
        df["rolling_away_points_5"]
    )

    return df

# Add rolling goal difference (total, home, away) over last N games per team.
def add_rolling_goal_diff_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()

    # Rolling total goal diff
    df["rolling_goal_diff_5"] = np.nan
    for team in df["team"].unique():
        team_mask = df["team"] == team
        team_goal_diff = df.loc[team_mask, "goal_diff"]

        df.loc[team_mask, "rolling_goal_diff_5"] = (team_goal_diff.shift(1).rolling(window).mean())

    # Rolling home goal diff
    df["rolling_home_goal_diff_5"] = np.nan
    for team in df["team"].unique():
        team_home_mask = (df["team"] == team) & (df["is_home"] == 1)
        team_goal_diff_home = df.loc[team_home_mask, "goal_diff"]

        df.loc[team_home_mask, "rolling_home_goal_diff_5"] = (team_goal_diff_home.shift(1).rolling(window).mean())

    # Rolling away goal diff
    df["rolling_away_goal_diff_5"] = np.nan
    for team in df["team"].unique():
        team_away_mask = (df["team"] == team) & (df["is_home"] == 0)
        team_goal_diff_away = df.loc[team_away_mask, "goal_diff"]

        df.loc[team_away_mask, "rolling_away_goal_diff_5"] = (team_goal_diff_away.shift(1).rolling(window).mean())

    return df

# Select final columns to keep for modeling.
def build_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "match_id",
        "season",
        "date",
        "team",
        "opponent",
        "is_home",
        "result",
        "goals_for",
        "goals_against",
        "points",
        "prob_book_home",
        "prob_book_draw",
        "prob_book_away",
        "goal_diff",
        "xg_diff",
        "xg_for",
        "xg_against",
        "rolling_xg_for_5",
        "rolling_xg_against_5",
        "rolling_xg_diff_5",
        "rolling_points_5",
        "rolling_home_points_5",
        "rolling_away_points_5",
        "strength_points_5",
        "rolling_goal_diff_5",
        "rolling_home_goal_diff_5",
        "rolling_away_goal_diff_5",
    ]

    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].copy()

# Save the final features dataset.
def save_features(df: pd.DataFrame, path: str = FEATURES_OUTPUT_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"  -> Saved features dataset to: {path}")

# Main function to execute the full feature engineering pipeline.
def run_feature_engineering(input_path: Path = MATCHES_LONG_PATH, output_path: Path = FEATURES_OUTPUT_PATH) -> pd.DataFrame:
    print("\n===== Engineering Features Pipeline (04) =====")
    
    # 1. Load data
    df_long = load_long_matches(input_path)

    # 2. Add basic differences
    df_long = add_basic_diff_columns(df_long)
    print("  -> Basic diff columns added (goal_diff, xg_diff).")

    # 3. Add points
    df_long = add_points_column(df_long)
    print("  -> Points column added.")

    # 4. Add rolling features (window=5)
    df_long = add_rolling_xg_features(df_long, window=5)
    df_long = add_rolling_points_features(df_long, window=5)
    df_long = add_rolling_goal_diff_features(df_long, window=5)
    print("  -> All rolling features added (xG, Points, Goal Diff).")

    # 5. Build final dataset
    df_features = build_model_dataset(df_long)
    print(f"  -> Final features shape: {df_features.shape}")

    # 6. Save features
    save_features(df_features, output_path)
    
    print("===== Feature Engineering Complete. ✅ =====\n")
    return df_features

if __name__ == "__main__":
    run_feature_engineering()

