# eLEKSYON ‘28: Voter Turnout Prediction System for the 2028 Philippine Elections

A Linear Algebra and Machine Learning pipeline that analyzes historical Philippine election data (2004–2025) across 55 cities and municipalities to forecast voter turnout for the 2028 elections.

## Variables
* **Inputs ($X$):**
  1. `previous_election_turnout_pct` — Historical Election Year Turnout (%)
  2. `registered_voters` — Total Registered Voters
  3. `voting_age_population` — Voting-Age Population (18+)
  4. `population_growth_rate_pct` — Population Growth Rate (%)
* **Target ($Y$):**
  * `actual_voter_turnout_pct` — Actual Voter Turnout (%)

## Methodology & Results
* **Matrix Diagnostics & Standardization:** Verified full column rank ($\text{rank} = 4$) and standardized predictors ($\mu = 0, \sigma = 1$).
* **PCA Dimensionality Reduction:** Compressed 4 features into $k = 3$ principal components retaining **99.61%** of total variance.
* **OLS Regression:** Achieved a test **RMSE of 2.63%** and **$R^2 = 0.674$** on unseen election data.

## System Features
1. **Historical Election Data Management:** Summarizes voter turnout and registration trends from 2004 to 2025.
2. **2028 Voter Turnout Prediction:** Generates municipal turnout estimates for 2028.
3. **Geographical Breakdown:** Aggregates predictions by region and city/municipality.
4. **Interactive Dashboard:** Streamlit web app displaying trend charts, regional/municipal filters, and PCA/OLS diagnostics.
