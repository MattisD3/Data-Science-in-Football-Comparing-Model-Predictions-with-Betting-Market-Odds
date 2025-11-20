import pandas as pd
import numpy as np
from pathlib import Path

# Elo parameters
BASE_RATING = 1500      # initial Elo for all teams
K_FACTOR = 25           # sensitivity of Elo to each match
HOME_ADVANTAGE = 80     # home-field advantage in Elo points

# Compute expected probability of a home win using Elo formula.
def expected_home_win(r_home: float, r_away: float, home_advantage: float=HOME_ADVANTAGE) -> float :
    rh = r_home + home_advantage
    ra = r_away
    return 1 / (1 + 10 ** ((ra - rh) / 400))

# Update a team's Elo rating based on match outcome.
def update_elo(rating: float, score: float, expected_score: float, k: float=K_FACTOR) -> float :
    return rating + k * (score - expected_score)

# Compute Elo ratings for a wide-format dataset (one row per match).
def compute_elo_wide(df: pd.DataFrame) -> pd.DataFrame:
    # Expected columns:
        # - match_id
        # - date
        # - home_team
        # - away_team
        # - result (H/D/A)
    
    df = df.sort_values(["date", "match_id"]).copy()

    ratings = {}
    elo_home_before = []
    elo_away_before = []

    for index, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        result = row["result"]

        r_home = ratings.get(home, BASE_RATING)
        r_away = ratings.get(away, BASE_RATING)

        # Save Elo BEFORE match
        elo_home_before.append(r_home)
        elo_away_before.append(r_away)

        # Expected probability for home team
        p_home = expected_home_win(r_home, r_away)

        # Actual score for home team
        if result == "H":
            s_home = 1.0
        elif result == "D":
            s_home = 0.5
        elif result == "A":
            s_home = 0.0
        else:
            raise ValueError(f"Unknown result value: {result}")

        # Update ratings
        new_r_home = update_elo(r_home, s_home, p_home)
        new_r_away = update_elo(r_away, 1 - s_home, 1 - p_home)

        ratings[home] = new_r_home
        ratings[away] = new_r_away

    # Add Elo features
    df["elo_home_before"] = elo_home_before
    df["elo_away_before"] = elo_away_before
    df["elo_diff_home"] = df["elo_home_before"] - df["elo_away_before"]

    return df

# Load wide match data, compute Elo, save minimal Elo dataset.
def compute_and_save_elo(input_path, output_path):

    df = pd.read_csv(input_path, parse_dates=["date"])
    df_elo = compute_elo_wide(df)

    # Keep only Elo-related columns
    elo_minimal = df_elo[[
        "match_id",
        "date",
        "home_team",
        "away_team",
        "result",
        "elo_home_before",
        "elo_away_before",
        "elo_diff_home"
    ]]

    elo_minimal.to_csv(output_path, index=False)

    print(f"Elo rating file saved to: {output_path}")


if __name__ == "__main__":
    compute_and_save_elo("data/processed/matches_wide_22_23.csv", "data/processed/elo_rating.csv")
