import pandas as pd

CSV_PATH = "data/raw/data_bookmakers 22-23.csv"
OUTPUT_PATH = "data/processed/clean_bookmakers_22_23.csv"

# Load raw data
def load_raw_data():
    df = pd.read_csv(CSV_PATH)
    print("Données chargées :", df.shape)
    return df

# Select only the relevant columns
def select_relevant_columns(df):
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
    df = df[cols]
    print("Colonnes sélectionnées :", df.shape)
    return df

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

def normalize_team_names(df):
    df["HomeTeam"] = df["HomeTeam"].replace(TEAM_MAPPING)
    df["AwayTeam"] = df["AwayTeam"].replace(TEAM_MAPPING)
    return df

# Clean the missing values
def clean_missing_values(df):
    df = df.dropna()
    print("Après nettoyage :", df.shape)
    return df

# Compute the average of the 3 bookmakers odds
def compute_avg_odds(df):
    df["home_odds"] = df[["B365H", "PSH", "WHH"]].mean(axis=1)
    df["draw_odds"] = df[["B365D", "PSD", "WHD"]].mean(axis=1)
    df["away_odds"] = df[["B365A", "PSA", "WHA"]].mean(axis=1)
    return df

# Convert odds to probabilities
def convert_odds_to_probabilities(df):
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
def standardize_columns(df):
    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
    })
    # Convert date to datetime (day/month/year)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return df


# Save clean data
def save_clean_data(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Données nettoyées sauvegardées dans {OUTPUT_PATH}")

# MAIN
if __name__ == "__main__":
    df = load_raw_data()
    df = select_relevant_columns(df)
    df = normalize_team_names(df)
    df = clean_missing_values(df)
    df = compute_avg_odds(df)
    df = convert_odds_to_probabilities(df)
    df = standardize_columns(df)
    # Keep only specific columns
    df = df[[
        "date",
        "home_team","away_team",
        "home_odds","draw_odds","away_odds",
        "prob_book_home","prob_book_draw","prob_book_away"
    ]]
    save_clean_data(df)
