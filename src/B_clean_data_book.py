import pandas as pd
from pathlib import Path

# Paths
RAW_BOOK_PATH = Path("data/raw/data_bookmakers_22_23.csv")
PROCESSED_BOOK_PATH = Path("data/processed/clean_bookmakers_22_23.csv")

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

# Load raw data
def load_raw_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

# Select only the relevant columns
def select_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "FTR", 
        # Bet365
        "B365H", "B365D", "B365A",
        # Pinnacle
        "PSH", "PSD", "PSA",
        # William Hill 
        "WHH", "WHD", "WHA"
    ]
    return df[cols].copy()

# Normalize team name variations 
def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df["HomeTeam"] = df["HomeTeam"].replace(TEAM_MAPPING)
    df["AwayTeam"] = df["AwayTeam"].replace(TEAM_MAPPING)
    return df

# Clean the missing values
def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()

# Compute the average of the 3 bookmakers odds
def compute_avg_odds(df: pd.DataFrame) -> pd.DataFrame:
    df["home_odds"] = df[["B365H", "PSH", "WHH"]].mean(axis=1)
    df["draw_odds"] = df[["B365D", "PSD", "WHD"]].mean(axis=1)
    df["away_odds"] = df[["B365A", "PSA", "WHA"]].mean(axis=1)
    return df

# Convert odds to probabilities
def convert_odds_to_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    # Convert odds to raw probabilities
    df["pH_raw"] = 1 / df["home_odds"]
    df["pD_raw"] = 1 / df["draw_odds"]
    df["pA_raw"] = 1 / df["away_odds"]

    # Sum to remove the bookmaker's margin
    total = df["pH_raw"] + df["pD_raw"] + df["pA_raw"]

    df["prob_book_home"] = df["pH_raw"] / total
    df["prob_book_draw"] = df["pD_raw"] / total
    df["prob_book_away"] = df["pA_raw"] / total

    return df

# Rename columns to standardized lowercase names
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
    })
    # Convert date to datetime (day/month/year)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return df

# Normalize team name variations 
def normalize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    df["HomeTeam"] = df["HomeTeam"].replace(TEAM_MAPPING)
    df["AwayTeam"] = df["AwayTeam"].replace(TEAM_MAPPING)
    return df

# Save clean data
def save_clean_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data → {path}")

# Execute full bookmakers cleaning pipeline and return cleaned DataFrame.
def run_cleaning_bookmakers_data(input_path: Path = RAW_BOOK_PATH, output_path: Path = PROCESSED_BOOK_PATH) -> pd.DataFrame:
    print("===== Cleaning bookmakers odds (02) =====")
    df = load_raw_data(input_path)
    print(f"Loaded raw data: {df.shape[0]} rows")

    df = select_relevant_columns(df)
    df = normalize_team_names(df)
    
    before = df.shape[0]
    df = clean_missing_values(df)
    after = df.shape[0]
    print(f"Dropped missing values: {before - after} rows removed")

    df = compute_avg_odds(df)
    df = convert_odds_to_probabilities(df)
    df = standardize_columns(df)

    df = df[
        [
            "date",
            "home_team", "away_team",
            "home_odds", "draw_odds", "away_odds",
            "prob_book_home", "prob_book_draw", "prob_book_away",
        ]
    ]

    print("Final bookmakers dataframe preview:")
    print(df.head(), "\n")

    save_clean_data(df, output_path)
    print("===== Cleaning complete. ✅ =====\n")

    return df


if __name__ == "__main__":
    run_cleaning_bookmakers_data()
