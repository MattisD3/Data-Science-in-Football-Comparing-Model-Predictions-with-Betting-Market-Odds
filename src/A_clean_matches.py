import pandas as pd
from pathlib import Path

# Paths
RAW_MATCHES_PATH = Path("data/raw/matches.csv")
PROCESSED_MATCHES_PATH = Path("data/processed/clean_matches_22_23.csv")

TEAM_MAPPING = {
    "Brighton": "Brighton & Hove Albion",
    "Brighton and Hove Albion": "Brighton & Hove Albion",

    "Leeds" : "Leeds United",
    "Leeds United" : "Leeds United",

    "Leicester" : "Leicester City",
    "Leicester City": "Leicester City",

    "Man City": "Manchester City",
    "Manchester City": "Manchester City",

    "Man United": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Manchester United": "Manchester United",

    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Newcastle United": "Newcastle United",

    "Nott'ham Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",

    "Sheffield Utd": "Sheffield United",
    "Sheffield United": "Sheffield United",

    "Tottenham": "Tottenham Hotspur",
    "Tottenham Hotspur": "Tottenham Hotspur",

    "West Brom": "West Bromwich Albion",
    "West Bromwich Albion": "West Bromwich Albion",

    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers",

    "West Ham": "West Ham United",
    "West Ham United": "West Ham United"
}

# Normalize team name variations for correct Home/Away merging
def normalize_team_names(df) -> pd.DataFrame:
    df["team"] = df["team"].replace(TEAM_MAPPING)
    df["opponent"] = df["opponent"].replace(TEAM_MAPPING)
    return df

# Load, clean, filter, and construct the final Premier League 22–23 match dataset.
def load_and_clean_matches(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # Premier League only
    df = df[df["comp"] == "Premier League"].copy()

    # Convert and filter by date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= "2022-08-01") & (df["date"] <= "2023-06-30")].copy()

    # Normalize team names
    df = normalize_team_names(df)

    # Home / Away split
    home_df = df[df["venue"] == "Home"].copy()
    away_df = df[df["venue"] == "Away"].copy()

    # Rename home columns
    home_df = home_df.rename(columns={
        "team": "home_team",
        "opponent": "away_team",
        "gf": "home_goals",
        "ga": "away_goals",
        "xg": "home_xg",
        "xga": "away_xg",
        "sh": "home_shots",
        "sot": "home_sot",
        "poss": "home_poss"
    })

    # Rename away columns
    away_df = away_df.rename(columns={
        "team": "away_team_check",
        "opponent": "home_team_check",
        "gf": "away_goals_check",
        "ga": "home_goals_check",
        "xg": "away_xg_check",
        "xga": "home_xg_check",
        "sh": "away_shots_check",
        "sot": "away_sot_check",
        "poss": "away_poss_check"
    })

    # Merge home + away rows
    merged = home_df.merge(
        away_df,
        left_on=["date", "home_team", "away_team"],
        right_on=["date", "home_team_check", "away_team_check"],
        how="inner"
        )

    # Build final dataframe
    final = merged[[
        "date",
        "home_team", "away_team",
        "home_goals", "away_goals",
        "home_xg", "away_xg",
        "home_shots", "away_shots_check",
        "home_sot", "away_sot_check",
        "home_poss", "away_poss_check"
        ]].copy()

    final = final.rename(columns={
        "away_shots_check": "away_shots",
        "away_sot_check": "away_sot",
        "away_poss_check": "away_poss"
        })

    # Remove duplicates
    final = final.drop_duplicates(subset=["date", "home_team", "away_team"])
    final = final.sort_values("date")

    return final

# Generate and save the cleaned match dataset.
def run_clean_matches(input_path: Path = RAW_MATCHES_PATH, output_path: Path = PROCESSED_MATCHES_PATH) -> None:
    print("===== Cleaning match data (01) =====")
    output_path.parent.mkdir(parents=True, exist_ok=True) 

    df = load_and_clean_matches(input_path) 
    df.to_csv(output_path, index=False)

    print(f"Clean matches saved to: {output_path}")
    print("===== Cleaning complete. ✅ =====\n")


if __name__ == "__main__":
    run_clean_matches()
