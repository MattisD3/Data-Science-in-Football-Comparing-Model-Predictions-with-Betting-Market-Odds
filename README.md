# Data Science in Football : Comparing Model Predictions with Betting Market Odds.

**Category** : Data Analysis & Visualization
**Name** : Descamps Mattis

## Problem statement or motivation
As a football fan and also a former gambler, I always wanted to know why sometimes the odds are so high for such big teams or so low for a small team. To understand that i will try to do my own odds based on football metrics such as `xG`, `shot_on_target`, `possession`, etc. by using machine learning models and then compare them to the one of the bookmakers. The aim of this project is to find out where and why a probability created from a machine learning model diverges from the one implied by the bookmakers. This project will also help me understand which statistics are the more "useful" to predict a winner.

My research question is :
***To what extent can football performance metrics such as expected goals (xG), shots, and possession be used to predict match outcome probabilities, and how do these model-implied probabilities differ from bookmaker odds?***

## Planned approach and technologies
1. **Data collection**
   I will use different sources such as:
   - [Understat](https://understat.com/) for the collection of the fooball metrics such as the xG, shots on target etc.
   - [Fbref](https://fbref.com/en/comps/9/Premier-League-Stats) also for the football metrics.
   - [Football-data.co.uk](https://www.football-data.co.uk/) for the bookmakers odds.
     
2. **Data cleaning/preparation**
   - Delete useless data then transform the data of each games into two lines (one for the home team and one for the away team).
   - Create new data from existing one such as: `xg_for_avg_last5`, `poss_diff`, `poss_diff_avg_last5` and more to indicates the trends, see the team recent form.
   - Merge with bookmakers odds and convert them to implied probabilities.
   - Standardize datasets using **pandas** and **NumPy**

3. **Modeling & evaluation**
   - Train classification models (*Multinomial Logistic Regression*, *Random Forest*, *XGBoost*) to estimate probabilities for home/draw/away outcomes.
   - Calibrate probabilities using *Platt* and *Isotonic scaling*.
   - Evaluate model performance using *Log Loss* and *Brier Score* to assess predictive quality and calibration.
   
4. **Visualization & Analysis**
   - Use **matplotlib** and **seaborn** to visualize the relationships and the results.
   - Visualization tools (**scatter plots**, **heatmaps**) will show where the model and bookies disagree and explore why (e.g., overreaction to recent form, missing crucial info such as best player being injured).
   - Visualization will support interpretability and make results more intuitive for non-technical readers.

## Expected challenges and how I’ll address them
   - Ensuring data consistency across my different sources.
   - Avoiding information leakage by using only pre-match features.
   - Handling correctly the different team names when merging data.
     
## Success criteria (how will I know it’s working?)   
   - The model produces calibrated and consistent data.
   - Visual and quantitative comparisons between model and bookmaker odds reveal systematic divergences.
   - The project provides clear explanations of why and where data-driven models and market expectations differ.

## Stretch goals (if time permits)
- Extend anaylsis accross multiple leagues.
- Backtesting : simulate a small theoretical betting strategy using model and bookmaker probabilities to test for potential value bets and analyze market mispricing.

# test update from VS Code3i
# double test
## test du Push