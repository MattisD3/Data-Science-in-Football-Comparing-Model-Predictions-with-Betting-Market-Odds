import pandas as pd
import numpy as np
from pathlib import Path

# Input path
MATCHES_WIDE_PATH = Path("data/processed/matches_wide_22_23.csv")
# Output path
ELO_OUTPUT_PATH = Path("data/processed/elo_ratings_22_23.csv")

# Elo parameters
BASE_RATING = 1500      # initial Elo for all teams
K_FACTOR = 25           # sensitivity of Elo to each match
HOME_ADVANTAGE = 80     # home-field advantage in Elo points

# Compute expected probability of a home win using Elo formula, adjusted for home advantage.
def expected_home_win(r_home: float, r_away: float, home_advantage: float=HOME_ADVANTAGE) -> float :
    rh = r_home + home_advantage  # Apply Home Advantage (adds points to the Home team's rating)
    ra = r_away
    return 1 / (1 + 10 ** ((ra - rh) / 400))

# Update a team's Elo rating based on match outcome.
def update_elo(rating: float, score: float, expected_score: float, k: float=K_FACTOR) -> float :
    return rating + k * (score - expected_score)

# Calculates Elo ratings for each team, match by match, on a wide-format DataFrame. The calculation is based on the perspective of the Home Team.
def compute_elo_wide(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure data is sorted chronologically    
    df = df.sort_values(["date", "match_id"]).copy()

    ratings = {}
    elo_home_before = []
    elo_away_before = []

    for index, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        result = row["result"]
        
        # Get current ratings (default to BASE_RATING if new team)
        r_home = ratings.get(home, BASE_RATING)
        r_away = ratings.get(away, BASE_RATING)

        # Save Elo BEFORE match
        elo_home_before.append(r_home)
        elo_away_before.append(r_away)

        # Expected probability for home team
        p_home = expected_home_win(r_home, r_away)
        p_away = 1 - p_home

        # Actual score for home team
        if result == "H":
            s_home = 1.0
        elif result == "D":
            s_home = 0.5
        elif result == "A":
            s_home = 0.0
        else:
            raise ValueError(f"Unknown result value: {result}")

        s_away = 1 - s_home # Actual score for away team

        # Update ratings
        new_r_home = update_elo(r_home, s_home, p_home)
        new_r_away = update_elo(r_away, s_away, p_away)

        ratings[home] = new_r_home
        ratings[away] = new_r_away

    # Add Elo features
    df["elo_home_before"] = elo_home_before
    df["elo_away_before"] = elo_away_before
    df["elo_diff_home"] = df["elo_home_before"] - df["elo_away_before"]

    return df

# Loads the wide-format match data, computes Elo ratings, and saves a minimal dataset containing only the Elo features (before match).
def run_elo_rating_wide(input_path: Path = MATCHES_WIDE_PATH, output_path: Path = ELO_OUTPUT_PATH) -> None:
    print("\n===== Starting ELO Rating Calculation (06) =====")

    # 1. Load data
    df = pd.read_csv(input_path, parse_dates=["date"])
    print(f"  -> Loaded wide match data: {df.shape}")

    # 2. Compute ELO
    df_elo = compute_elo_wide(df)

    # 3. Keep only Elo-related columns for minimal output
    elo_minimal = df_elo[[
        "match_id",
        "date",
        "home_team", # Use HomeTeam/AwayTeam for consistency with wide format
        "away_team",
        "result",
        "elo_home_before",
        "elo_away_before",
        "elo_diff_home"
        ]]

    # 4. Save the file
    elo_minimal.to_csv(output_path, index=False)
    print(f"  -> ELO rating file saved to: {output_path}")
    print(f"  -> Final shape of ELO data: {elo_minimal.shape}")
    print("===== ELO Calculation Complete. ✅ =====")


if __name__ == "__main__":
    run_elo_rating_wide()