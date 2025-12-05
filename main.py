# Import the function to clean the match data match 
from src.A_clean_matches import run_clean_matches

# Import the function to clean the bookmakers data
from src.B_clean_data_book import run_cleaning_bookmakers_data

# Import the function to create rolling features and final dataset
from src.C_merge_data import run_merge_data

# Import the function to create rolling features and final dataset
from src.D_feature_engineering import run_feature_engineering

# Import the function to run the baseline logistic regression
from src.E_model_baseline_logistic import run_logistic_baseline

# Import the function to compute the elo
from src.F_elo_rating import run_elo_rating_wide

# Import the function to merge the Elo to the previous rolling features
from src.G_add_elo_to_long_features import run_add_elo_features

# Import the function to run the baseline logistic regression WITH ELO
from src.H_model_baseline_logistic_with_elo import run_baseline_with_elo

# Import the function to run the Random Forest Model
from src.I_model_random_forest import run_random_forest_pipeline

# Import the function to run the XG Boost Model
from src.J_model_xgboost import run_xgboost_pipeline

# Import the function to Calibrate the 3 model 
from src.K_calibration_experiments import run_full_calibration_pipeline

# Import the function to compare my best model to the bookmakers
from src.L_model_vs_bookmakers import run_comparison

if __name__ == "__main__":
    run_clean_matches()
    run_cleaning_bookmakers_data()
    run_merge_data()
    run_feature_engineering()
    run_logistic_baseline()
    run_elo_rating_wide()
    run_add_elo_features()
    run_baseline_with_elo()
    print("\n ********** NOW EVERYTHING IS WITH THE ELO AS A FEATURE. ********** ")
    run_random_forest_pipeline()
    run_xgboost_pipeline()
    run_full_calibration_pipeline()
    run_comparison()
