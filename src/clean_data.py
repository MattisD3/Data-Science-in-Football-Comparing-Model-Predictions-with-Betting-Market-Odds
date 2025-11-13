import pandas as pd

CSV_PATH = "data/data bookmakers 22-23.csv"
OUTPUT_PATH = "data/clean_bookmakers_22_23.csv"

# Charger les données brutes
def load_raw_data():
    df = pd.read_csv(CSV_PATH)
    print("Données chargées :", df.shape)
    return df

# Sélectionner seulement les colonnes utiles
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

# Nettoyer les valeurs manquantes
def clean_missing_values(df):
    df = df.dropna()
    print("Après nettoyage :", df.shape)
    return df

# Calculer la moyenne des 3 bookmakers
def compute_avg_odds(df):
    df["AvgH"] = df[["B365H", "PSH", "WHH"]].mean(axis=1)
    df["AvgD"] = df[["B365D", "PSD", "WHD"]].mean(axis=1)
    df["AvgA"] = df[["B365A", "PSA", "WHA"]].mean(axis=1)
    return df

# Transformer les cotes en proba
def convert_odds_to_probabilities(df):
    # Convert odds to raw probabilities
    df["pH_raw"] = 1 / df["AvgH"]
    df["pD_raw"] = 1 / df["AvgD"]
    df["pA_raw"] = 1 / df["AvgA"]

    # Somme pour enlever la marge
    total = df["pH_raw"] + df["pD_raw"] + df["pA_raw"]

    df["ProbH"] = df["pH_raw"] / total
    df["ProbD"] = df["pD_raw"] / total
    df["ProbA"] = df["pA_raw"] / total

    return df

# Sauvegarder les données propres
def save_clean_data(df):
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Données nettoyées sauvegardées dans {OUTPUT_PATH}")

# MAIN
if __name__ == "__main__":
    df = load_raw_data()
    df = select_relevant_columns(df)
    df = clean_missing_values(df)
    df = compute_avg_odds(df)
    df = convert_odds_to_probabilities(df)
    save_clean_data(df)
