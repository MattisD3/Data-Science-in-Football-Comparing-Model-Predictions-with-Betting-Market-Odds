import pandas as pd
from pathlib import Path

# Input paths
MATCHES_PATH = Path("data/processed/clean_matches_22_23.csv")
BOOKMAKERS_PATH = Path("data/processed/clean_bookmakers_22_23.csv")
# Output paths
MATCHES_WIDE_PATH = Path("data/processed/matches_wide_22_23.csv")
MATCHES_LONG_PATH = Path("data/processed/matches_long_22_23.csv")

# Load the cleaned data.
def load_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    matches_df = pd.read_csv(MATCHES_PATH)
    bookmakers_df = pd.read_csv(BOOKMAKERS_PATH)

    return matches_df, bookmakers_df

# Merge match stats with bookmaker odds on (date, home_team, away_team).
def merge_matches_and_odds(matches_df: pd.DataFrame, bookmakers_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = matches_df.merge(
        bookmakers_df,
        on=["date", "home_team", "away_team"],
        how="inner",
        validate="one_to_one",
        )
    return merged_df

# Add match result column: H (home win), D (draw), A (away win).
def add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    def compute_result(row):
        if row["home_goals"] > row["away_goals"]:
            return "H"
        elif row["home_goals"] < row["away_goals"]:
            return "A"
        else:
            return "D"
        
    df = df.copy()
    df["result"] = df.apply(compute_result, axis=1)
    return df

# Select wide-format match-level columns and add match_id and season.
def build_matches_wide(merged_df: pd.DataFrame) -> pd.DataFrame:
    matches_wide = merged_df[[
        "date",
        "home_team", "away_team",
        "result",
        "home_goals", "away_goals",
        "home_xg", "away_xg",
        "home_shots", "away_shots",
        "home_sot", "away_sot",
        "home_poss", "away_poss",
        "home_odds", "draw_odds", "away_odds",
        "prob_book_home", "prob_book_draw", "prob_book_away"
        ]].copy()

    # Unique id per match
    matches_wide["match_id"] = range(1, len(matches_wide) + 1)
    matches_wide["season"] = "2022-2023"

    # Reorder columns (optional)
    matches_wide = matches_wide[[
        "match_id", "season", "date",
        "home_team", "away_team",
        "result",
        "home_goals", "away_goals",
        "home_xg", "away_xg",
        "home_shots", "away_shots",
        "home_sot", "away_sot",
        "home_poss", "away_poss",
        "home_odds", "draw_odds", "away_odds",
        "prob_book_home", "prob_book_draw", "prob_book_away",
    ]]

    return matches_wide

# Convert wide-format matches (1 row per match) to long format (2 rows per match: one for each team).
def build_matches_long(matches_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for index, row in matches_wide.iterrows():
        # Home team row (perspective of the home team)
        rows.append({
            "match_id": row["match_id"],
            "season": row["season"],
            "date": row["date"],
            "team": row["home_team"],
            "opponent": row["away_team"],
            "is_home": 1,
            "goals_for": row["home_goals"],
            "goals_against": row["away_goals"],
            "xg_for": row["home_xg"],
            "xg_against": row["away_xg"],
            "shots_on_target_for": row["home_sot"],
            "shots_on_target_against": row["away_sot"],
            "poss": row["home_poss"],
            "prob_book_home": row["prob_book_home"],
            "prob_book_draw": row["prob_book_draw"],
            "prob_book_away": row["prob_book_away"],
            "result": row["result"],
        })

        # Away team row (perspective of the away team)
        rows.append({
            "match_id": row["match_id"],
            "season": row["season"],
            "date": row["date"],
            "team": row["away_team"],
            "opponent": row["home_team"],
            "is_home": 0,
            "goals_for": row["away_goals"],
            "goals_against": row["home_goals"],
            "xg_for": row["away_xg"],
            "xg_against": row["home_xg"],
            "shots_on_target_for": row["away_sot"],
            "shots_on_target_against": row["home_sot"],
            "poss": row["away_poss"],
            "prob_book_home": row["prob_book_home"],
            "prob_book_draw": row["prob_book_draw"],
            "prob_book_away": row["prob_book_away"],
            "result": row["result"],
        })

    matches_long = pd.DataFrame(rows)
    matches_long = matches_long.sort_values(["date", "match_id", "team"]).reset_index(drop=True)
    return matches_long

# Execute full merging data pipeline and return cleaned DataFrame.
def run_merge_data(wide_path: Path = MATCHES_WIDE_PATH, long_path: Path = MATCHES_LONG_PATH) -> None:
    print("\n===== Loading cleaned data (03) =====")
    # 1. Load cleaned data
    matches_df, bookmakers_df = load_clean_data()
    print(f"  -> Matches loaded: {matches_df.shape} | Bookmakers loaded: {bookmakers_df.shape}") # Must be (380, 13) | (380,9)

    # 2. Merge dataframes
    merged_df = merge_matches_and_odds(matches_df, bookmakers_df)
    print("  -> Merge completed.")
    print(f"  -> Merged shape: {merged_df.shape} (3 columns used for merge)") # Must be (380, 19) (19= 13 + 9 - 3 common columns) (result, match_id, season)

    # 3. Add result column
    merged_df = add_result_column(merged_df)

    # 4. Build wide format (one row per match)
    matches_wide = build_matches_wide(merged_df)
    print(f"  -> matches_wide shape: {matches_wide.shape}") # Must be (380, 22) (19+3)

    # 5. Build long format (two rows per match, one for each team)
    matches_long = build_matches_long(matches_wide)
    print(f"  -> matches_long shape: {matches_long.shape}") # Must be (760, 17)

    # 6. Save outputs
    wide_path.parent.mkdir(parents=True, exist_ok=True) 
    
    matches_wide.to_csv(wide_path, index=False)
    matches_long.to_csv(long_path, index=False)

    print(f"  -> Saved wide format to: {wide_path}")
    print(f"  -> Saved long format: {long_path}")
    print("===== Merge Complete. ✅ =====\n")

if __name__ == "__main__":
    run_merge_data(MATCHES_WIDE_PATH, MATCHES_LONG_PATH)