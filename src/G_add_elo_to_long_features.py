import pandas as pd
import numpy as np
from pathlib import Path

# Input paths
FEATURES_LONG_PATH = Path("data/processed/features_matches_long_22_23.csv")
ELO_OUTPUT_PATH = Path("data/processed/elo_ratings_22_23.csv")
# Output path
FEATURES_ELO_OUTPUT_PATH = Path("data/processed/features_matches_long_elo_22_23.csv")

# Load engineered long-format match features (two rows per match, one per team).
def load_long_features(path: Path = FEATURES_LONG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"  -> Loaded long features (input for merge): {df.shape}")
    return df

# Load match-level Elo ratings (wide format: one row per match).
def load_elo(path: Path = ELO_OUTPUT_PATH) -> pd.DataFrame:
    df_elo = pd.read_csv(path, parse_dates=["date"])
    print(f"  -> Loaded Elo ratings (input for merge): {df_elo.shape}")
    return df_elo

# Merge match-level Elo ratings into the long-format dataset and build team-level Elo features.
def add_elo_to_long(df_long: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
    
    # Merge on match_id to bring home/away Elo
    merged = df_long.merge(
        df_elo[[
            "match_id",
            "elo_home_before",
            "elo_away_before",
            "elo_diff_home",
        ]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    # Sanity check
    if merged["elo_home_before"].isna().any():
        print(f"  -> WARNING: Some rows have missing Elo after merge (check match_id consistency).")

    # Build team-level Elo features
    is_home = merged["is_home"] == 1

    # Team Elo before the match (depends on is_home)
    merged["elo_team_before"] = np.where(is_home, merged["elo_home_before"], merged["elo_away_before"])

    # Opponent Elo before the match
    merged["elo_opponent_before"] = np.where(is_home, merged["elo_away_before"], merged["elo_home_before"])

    # Elo difference from the team's point of view
    merged["elo_diff_for_team"] = (merged["elo_team_before"] - merged["elo_opponent_before"])

    print(f"  -> Merged dataset shape: {merged.shape}")
    return merged

# Save the long-format dataset enriched with Elo features.
def save_long_with_elo(df: pd.DataFrame, path: Path = FEATURES_ELO_OUTPUT_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"  -> Saved long + Elo features dataset to: {path}")

# Main function to orchestrate the loading of features and Elo ratings, the merge operation and the final saving.
def run_add_elo_features(features_path: Path = FEATURES_LONG_PATH, elo_path: Path = ELO_OUTPUT_PATH, output_path: Path = FEATURES_ELO_OUTPUT_PATH) -> None:
    print("\n===== Starting ELO Feature Merge (07) =====")
    
    df_long = load_long_features(features_path)
    df_elo = load_elo(elo_path)

    df_with_elo = add_elo_to_long(df_long, df_elo)
    print("  ----- Final features dataframe preview -----")
    print(df_with_elo.head(), "\n")
    save_long_with_elo(df_with_elo, output_path)
    print("===== ELO Feature Merge Complete. ✅ =====\n")

if __name__ == "__main__":
    run_add_elo_features()