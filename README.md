# ⚽ FIFA World Cup 2026 Prediction System

## Overview

This project is a Machine Learning application developed to predict FIFA World Cup 2026 group-stage match results and standings.

The system uses historical international football match data from 2010 onwards and applies a Gradient Boosting Regression model to estimate the number of goals scored by each team. Predicted results are then used to generate group standings and determine qualified teams.

The application is built with Streamlit and provides an interactive dashboard for exploring predictions and tournament outcomes.

---

## Features

- Predict FIFA World Cup 2026 group-stage match scores
- Display real and predicted match results
- Generate group standings automatically
- Determine qualified teams based on tournament rules
- Interactive Streamlit dashboard
- Advanced feature engineering based on team performance
- Integration with football-data.org API for FIFA World Cup information

---

## Machine Learning Model

### Algorithm
- Gradient Boosting Regressor

### Training Data
- Historical international football matches (2010–2026)
- FIFA World Cup matches
- Continental tournaments
- Qualification matches
- Friendly matches

### Features Used

- Recent form points
- Average goals scored
- Average goals conceded
- Win rate
- Head-to-head statistics
- Neutral venue indicator
- Tournament importance weighting

---

## Data Sources

### Historical Dataset

- Historical international football match results stored in `results.csv`
- Used for training the machine learning model

### Football API

The project uses the football-data.org API to retrieve official FIFA World Cup information such as:

- Groups
- Teams
- Fixtures
- Match metadata

API Documentation:

https://www.football-data.org/documentation/quickstart

---

## Data Collection Pipeline

1. Historical football match data is loaded from `results.csv`.
2. FIFA World Cup information is collected using the football-data.org API.
3. Match features are generated:
   - Team form
   - Goals scored and conceded
   - Win rate
   - Head-to-head performance
4. The Gradient Boosting model predicts goals for each team.
5. Predictions are used to generate standings and qualification scenarios.

---


## Project Structure

```text
FIFA26/
│
├── app.py
├── results.csv
├── Fifa26_Phase1.txt
├── requirements.txt
├── README.md
│
└── 
```

## Installation

### Clone the repository

```bash
git https://github.com/hamdikbell/Fifa-world-cup-2026-prediction.git
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
cd FIFA2026
streamlit run app.py
```

---

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Gradient Boosting Regressor
- Requests
- Football-Data.org API

---

## Dashboard Sections

### 1. Group Standings
Displays rankings for all World Cup groups based on real and predicted results.

### 2. Scores & Results
Shows actual scores when available and model predictions for future matches.

### 3. Qualified Teams
Identifies teams advancing to the knockout stage according to tournament rules.

---

## Author

### Ikbel Hamdi
**Data Science Student**

---

## Disclaimer

This project is intended for educational and research purposes only.

Match predictions are generated using machine learning techniques and statistical analysis. Actual tournament outcomes may differ from the predicted results.

---



⭐ If you like this project, don't forget to give it a star on GitHub.