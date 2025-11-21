import pandas as pd
import numpy as np
from pathlib import Path

# Paths
FEATURES_LONG_PATH = Path("data/processed/features_matches_long_22_23.csv")
ELO_PATH = Path("data/processed/elo_rating_22_23.csv")
OUTPUT_PATH = Path("data/processed/features_matches_long_elo_22_23.csv")

# Load engineered long-format match features (two rows per match, one per team).
def load_long_features(path: Path = FEATURES_LONG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded long features: {df.shape}")
    return df

# Load match-level Elo ratings (wide format: one row per match).
def load_elo(path: Path = ELO_PATH) -> pd.DataFrame:
    df_elo = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded Elo ratings: {df_elo.shape}")
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
        print("Warning: some rows have missing Elo after merge.")

    # Build team-level Elo features
    is_home = merged["is_home"] == 1

    # Team Elo before the match (depends on is_home)
    merged["elo_team_before"] = np.where(is_home, merged["elo_home_before"], merged["elo_away_before"])

    # Opponent Elo before the match
    merged["elo_opponent_before"] = np.where(is_home, merged["elo_away_before"], merged["elo_home_before"])

    # Elo difference from the team's point of view
    merged["elo_diff_for_team"] = (merged["elo_team_before"] - merged["elo_opponent_before"])

    print("Long dataset with Elo features:", merged.shape)
    return merged

# Save the long-format dataset enriched with Elo features.
def save_long_with_elo(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    df.to_csv(path, index=False)
    print(f"Saved long + Elo features to: {path}")


def main():
    df_long = load_long_features(FEATURES_LONG_PATH)
    df_elo = load_elo(ELO_PATH)

    df_with_elo = add_elo_to_long(df_long, df_elo)
    save_long_with_elo(df_with_elo, OUTPUT_PATH)


if __name__ == "__main__":
    main()
