Sources of Data

This document lists the official sources used to obtain all datasets for the project, ensuring transparency and reproducibility.

 1. Bookmaker Odds Data 
    Source: Football-Data.co.uk
    Main page: [https://www.football-data.co.uk/englandm.php]
    File used: Premier League 2022–2023, renamed locally as: 
    
    data/data bookmakers 22-23.csv
    
    Columns used:
    - Match result: FTHG, FTAG, FTR
    - Bet365 odds: B365H, B365D, B365A
    - Pinnacle odds: PSH, PSD, PSA
    - William Hill odds: WHH, WHD, WHA

    Cleaning steps applied:
    - Selection of relevant columns
    - Removal of rows with missing values

    Computation of averaged odds:
    - AvgH, AvgD, AvgA

    Conversion of odds into probabilities (market-adjusted):
    - ProbH, ProbD, ProbA

    Export of cleaned dataset:
    - data/clean_bookmakers_22_23.csv

2. Match Statistics Data (xG, shots, possession, etc.)
    Source: Kaggle
    Dataset: Premier League Matches (2020–2024)
    URL: [https://www.kaggle.com/datasets/mhmdkardosha/premier-league-matches]

    Columns intended for use:
    - Match information: date, home_team, away_team
    - Expected goals: home_xg, away_xg
    - Possession: home_possession, away_possession
    - Shots: home_shots, away_shots

Additional advanced metrics where available

Planned cleaning steps:

Filter only the 2022–2023 Premier League season

Standardize team names to match bookmaker dataset

Merge with bookmaker data using:

Date

HomeTeam

AwayTeam

Export (planned):

data/clean_match_stats_22_23.csv

3. Final Dataset for Modeling

The final dataset (after merging bookmakers + match stats):

data/model_dataset_22_23.csv


It will include:

Actual match outcomes

Bookmakers' probabilities

Advanced match statistics

One row per match