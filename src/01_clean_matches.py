import pandas as pd

def load_and_clean_matches(path="data/raw/matches.csv"):
    df = pd.read_csv(path)

    # Premier League only
    df = df[df["comp"] == "Premier League"].copy()

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Season 2022-2023
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

    # Sort
    final = final.sort_values("date")

    return final

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
def normalize_team_names(df):
    df["team"] = df["team"].replace(TEAM_MAPPING)
    df["opponent"] = df["opponent"].replace(TEAM_MAPPING)
    return df


if __name__ == "__main__":
    df = load_and_clean_matches()
    df.to_csv("data/processed/clean_matches_22_23.csv", index=False)
    print("clean_matches_22_23.csv created.")
