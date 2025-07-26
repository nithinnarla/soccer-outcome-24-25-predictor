# ⚽ Soccer Match Outcome Predictor (Premier League 2024/25)

This repository contains a complete machine–learning project for predicting the
outcome of soccer matches in the **English Premier League 2024/25 season**.  The
project uses only free, publicly available data and tools.  It demonstrates
how to engineer features from raw match results, train multiple models,
evaluate them, and serve the best model via a Streamlit web application.

## 📊 Dataset

The raw data comes from the open‐source [`openfootball/england`][openfootball]
repository, which publishes match schedules and results in plain text for each
season.  For the 2024/25 season there are **20 teams** and **380 matches**
played between **16 August 2024** and **25 May 2025**【388831341396613†L0-L5】.  Each record in the
dataset includes the match number, round number, date and time, location,
home team, away team and final score (e.g. `"2 - 1"`).  A sample of the
raw season file shows the format:

> = English Premier League 2024/25
> 
> # Date       Fri Aug/16 2024 – Sun May/25 2025
> 
> # Teams      20
> 
> # Matches    380【388831341396613†L0-L5】

The CSV version of this schedule (`data/premier_league_2024_2025.csv`) is
provided in this repository.  It contains seven columns: `Match Number`,
`Round Number`, `Date`, `Location`, `Home Team`, `Away Team` and `Result`.

## 🔧 Feature Engineering

The predictive power of our model comes from engineered features that capture
team strength and recent form.  We compute the following features:

| Feature                     | Description |
|----------------------------|-------------|
| **elo_diff**               | Difference between the home and away Elo ratings.  Elo ratings start at 1 500 for each team and are updated after every match using a K‑factor of 30. |
| **home_form_points_avg**   | Average points (win = 3, draw = 1, loss = 0) earned by the home team over its last five games. |
| **away_form_points_avg**   | Average points earned by the away team over its last five games. |
| **home_goals_avg**         | Average goals scored by the home team over its last five games. |
| **away_goals_avg**         | Average goals scored by the away team over its last five games. |
| **home_conceded_avg**      | Average goals conceded by the home team over its last five games. |
| **away_conceded_avg**      | Average goals conceded by the away team over its last five games. |
| **rest_days_home**         | Days since the home team last played.  Missing values (start of season) are filled with the median rest period. |
| **rest_days_away**         | Days since the away team last played (median‑filled if missing). |
| **round_number**           | The round in which the match takes place. |

The target variable (`target`) is encoded as **0** for an away win, **1** for a draw
and **2** for a home win.

## 🤖 Model Training

The training script (`src/train_model.py`) performs the following steps:

1. **Load data** – reads the CSV file and invokes `create_feature_df` from
   `src/data_preprocessing` to generate features and the target.
2. **Split data** – uses an 80/20 train/test split with stratification by
   outcome.
3. **Scale features** – standardizes all numerical features with
   `StandardScaler`.
4. **Train models** – evaluates logistic regression, random forest and
   XGBoost (if available) via 5‑fold cross‑validation using macro F1 score.
5. **Select best model** – retrains the best‑performing model on the full
   training set and reports metrics on the held‑out test set.
6. **Save bundle** – saves the model, scaler, feature names and label map to
   `models/soccer_outcome_model.pkl` using `joblib`.

## 🌐 Streamlit App

The front‑end app (`app.py`) loads the trained model and provides an
interactive interface:

* **Predict tab** – choose any home and away team and see the predicted
  probabilities for a home win, draw or away win.  Features are computed
  on‑the‑fly from the final Elo and form statistics of the 2024/25 season.
* **Data Overview tab** – explore the raw dataset, view the distribution of
  match outcomes and see a correlation heatmap of the engineered features.
* **Feature Importance tab** – visualize which features contribute most to
  the model’s predictions using SHAP values (or fall back to feature
  importances / coefficients if SHAP is unavailable).

## 📂 Repository Structure

```
├── data
│   └── premier_league_2024_2025.csv   # Raw match results (20 teams, 380 matches)
├── models
│   └── soccer_outcome_model.pkl       # Saved model bundle (generated after training)
├── notebooks
│   └── eda.ipynb                      # Exploratory data analysis
├── src
│   ├── __init__.py
│   ├── data_preprocessing.py          # Feature engineering logic
│   └── train_model.py                 # Model training script
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Running Locally

1. **Clone the repository**

       git clone https://github.com/your‑username/soccer‑outcome‑predictor.git
       cd soccer‑outcome‑predictor

2. **Create a virtual environment and install dependencies**

       python3 -m venv .venv
       source .venv/bin/activate
       pip install --upgrade pip
       pip install -r requirements.txt

3. **Train the model (optional)** – if you wish to retrain the model or tweak
   hyperparameters, run:

       python -m src.train_model

   This will generate `models/soccer_outcome_model.pkl`.

4. **Run the Streamlit app**

       streamlit run app.py

   Navigate to the provided local URL to interact with the predictor.

## ☁️ Deploying on Streamlit Cloud (Free Tier)

1. **Create a GitHub repository** – push this project to a public GitHub
   repository.  Streamlit Cloud reads directly from GitHub.
2. **Sign in to Streamlit Cloud** – go to <https://streamlit.io/cloud> and sign
   in with your GitHub account.
3. **Deploy new app** – click “New app”, select your repository and branch,
   and specify `app.py` as the entrypoint.
4. **Requirements file** – Streamlit Cloud automatically installs packages
   from `requirements.txt` on the free tier.  No additional configuration is
   required.
5. **(Optional) Secrets** – if your model depends on private keys or
   credentials, add them via Streamlit’s secrets manager.  This project
   requires none.

After deployment Streamlit will build the app and provide a shareable URL.

## 📄 License

This project is released under the MIT License.  See [`LICENSE`](LICENSE) for
details.

[openfootball]: https://github.com/openfootball/england