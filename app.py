"""
Streamlit application for predicting the outcome of Premier League matches using
a machine learning model trained on the 2024/25 season data.  The app allows
users to select any two teams and outputs the predicted probabilities for a
home win, draw or away win.  Additional tabs provide a glimpse of the raw
match data, engineered features and model interpretability via SHAP feature
importance.

To run locally::

    streamlit run app.py

Make sure you have first trained the model by running ``python -m src.train_model``.
The trained model will be stored in ``models/soccer_outcome_model.pkl``.
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from datetime import datetime
from typing import Dict, List, Tuple

from src.data_preprocessing import parse_score  # reuse helper for score parsing


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    """Load the raw match results from the data folder."""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'premier_league_2024_2025.csv')
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M", errors='coerce')
    # Parse goals
    df[['home_goals', 'away_goals']] = df['Result'].str.split('-', expand=True).apply(lambda x: x.str.strip()).astype(int)
    return df


@st.cache_data
def compute_team_stats(df: pd.DataFrame) -> Dict[str, Dict[str, any]]:
    """Compute final Elo ratings and histories for each team after the season.

    Returns a dictionary keyed by team containing elo rating, lists of recent
    results, goals for, goals against and last played date.  This allows
    feature generation for arbitrary pairings at prediction time.
    """
    teams = pd.concat([df['Home Team'], df['Away Team']]).unique()
    elo = {team: 1500.0 for team in teams}
    results_hist = {team: [] for team in teams}
    gf_hist = {team: [] for team in teams}
    ga_hist = {team: [] for team in teams}
    last_played = {team: None for team in teams}
    last_round = int(df['Round Number'].max())
    # Sort by date to update sequentially
    for _, row in df.sort_values('Date').iterrows():
        home, away = row['Home Team'], row['Away Team']
        home_goals, away_goals = row['home_goals'], row['away_goals']
        # Determine result from home perspective
        if home_goals > away_goals:
            outcome = 1.0
        elif home_goals == away_goals:
            outcome = 0.5
        else:
            outcome = 0.0
        # Update histories before Elo update (to avoid using current match)
        results_hist[home].append(outcome)
        results_hist[away].append(1.0 - outcome)
        gf_hist[home].append(home_goals)
        ga_hist[home].append(away_goals)
        gf_hist[away].append(away_goals)
        ga_hist[away].append(home_goals)
        last_played[home] = row['Date']
        last_played[away] = row['Date']
        # Update Elo
        expected_home = 1.0 / (1.0 + 10 ** (-(elo[home] - elo[away]) / 400))
        new_home = elo[home] + 30.0 * (outcome - expected_home)
        new_away = elo[away] + 30.0 * ((1.0 - outcome) - (1.0 - expected_home))
        elo[home], elo[away] = new_home, new_away
    # Compile stats
    stats = {}
    for team in teams:
        stats[team] = {
            'elo': elo[team],
            'results': results_hist[team],
            'gf': gf_hist[team],
            'ga': ga_hist[team],
            'last_played': last_played[team]
        }
    stats['last_round'] = last_round
    # Provide median rest days to fill missing values later
    # Compute differences for each team except first match
    rest_values = []
    for team in teams:
        dates = [d for d in df[df['Home Team'] == team]['Date']] + \
                [d for d in df[df['Away Team'] == team]['Date']]
        dates = sorted(dates)
        for i in range(1, len(dates)):
            rest_values.append((dates[i] - dates[i-1]).days)
    if rest_values:
        stats['rest_median'] = float(np.median(rest_values))
    else:
        stats['rest_median'] = 7.0
    return stats


def get_features_for_match(home_team: str, away_team: str, stats: Dict[str, any]) -> np.ndarray:
    """Generate feature vector for a hypothetical match between two teams.

    Uses the final Elo and form statistics of each team based on 2024/25 season
    results to compute the same features used during training.
    """
    # Elo difference
    elo_diff = stats[home_team]['elo'] - stats[away_team]['elo']
    # Form points average (last 5 games)
    def form_points(lst: List[float]) -> float:
        if not lst:
            return np.nan
        pts = [3 * r for r in lst]  # r is 1.0/0.5/0.0; convert to 3/1/0
        return np.mean(pts[-5:])
    home_form = form_points(stats[home_team]['results'])
    away_form = form_points(stats[away_team]['results'])
    # Goals averages
    def avg_last(lst: List[int]) -> float:
        return np.nan if not lst else np.mean(lst[-5:])
    home_goals_avg = avg_last(stats[home_team]['gf'])
    away_goals_avg = avg_last(stats[away_team]['gf'])
    home_conceded_avg = avg_last(stats[home_team]['ga'])
    away_conceded_avg = avg_last(stats[away_team]['ga'])
    # Rest days for both teams.  We cannot derive a meaningful rest period from
    # the final season statistics because only the last match date is stored.
    # Instead we fall back to the median rest period computed across the season.
    rest_home = stats['rest_median']
    rest_away = stats['rest_median']
    # Round number (set to last_round + 1 for hypothetical next match)
    round_number = float(stats['last_round'] + 1)
    features = np.array([
        elo_diff,
        home_form,
        away_form,
        home_goals_avg,
        away_goals_avg,
        home_conceded_avg,
        away_conceded_avg,
        rest_home,
        rest_away,
        round_number
    ], dtype=float)
    # Replace any nan with median values of training features (approx via rest median)
    nan_mask = np.isnan(features)
    if nan_mask.any():
        # fill with median rest days; our preprocessing fills missing with column medians
        features[nan_mask] = stats['rest_median']
    return features


@st.cache_resource
def load_model() -> Dict[str, any]:
    """Load the trained model bundle from disk."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'soccer_outcome_model.pkl')
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def main():
    st.set_page_config(page_title="Soccer Outcome Predictor", layout="wide")
    st.title("⚽ Soccer Match Outcome Predictor (Premier League 2024/25)")
    # Load data and model
    df = load_raw_data()
    stats = compute_team_stats(df)
    model_bundle = load_model()
    if model_bundle is None:
        st.error("Model file not found. Please run `python -m src.train_model` first.")
        return
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_names = model_bundle['feature_names']
    label_map = model_bundle['label_map']

    # Tabs for navigation
    tabs = st.tabs(["Predict", "Data Overview", "Feature Importance"])
    # Predict tab
    with tabs[0]:
        st.header("Predict a match outcome")
        teams = sorted(df['Home Team'].unique())
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Select Home Team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
        with col2:
            away_team = st.selectbox("Select Away Team", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)
        if home_team == away_team:
            st.warning("Home and away teams must be different.")
        else:
            if st.button("Predict Outcome"):
                # Compute feature vector
                feats = get_features_for_match(home_team, away_team, stats)
                feats_scaled = scaler.transform(feats.reshape(1, -1))
                probs = model.predict_proba(feats_scaled)[0]
                pred_idx = int(np.argmax(probs))
                pred_label = label_map[pred_idx]
                outcome_mapping = {2: "Home Win", 1: "Draw", 0: "Away Win"}
                outcome_str = outcome_mapping[pred_idx]
                st.subheader(f"Prediction: {outcome_str}")
                st.write("Probabilities:")
                st.write({
                    "Home Win": round(float(probs[2]), 3),
                    "Draw": round(float(probs[1]), 3),
                    "Away Win": round(float(probs[0]), 3)
                })

    # Data Overview tab
    with tabs[1]:
        st.header("Data Overview and EDA")
        st.write("First 10 rows of the raw dataset:")
        st.dataframe(df.head(10))
        # Show distribution of match results
        result_counts = df['Result'].apply(lambda x: parse_score(x)[0] - parse_score(x)[1])
        # Map difference to label
        outcome_map = result_counts.apply(lambda x: 'Home Win' if x > 0 else ('Draw' if x == 0 else 'Away Win'))
        st.write("Distribution of outcomes:")
        st.bar_chart(outcome_map.value_counts())
        # Correlation heatmap of engineered features
        st.write("Correlation heatmap of engineered features:")
        # Create feature df for correlation only once
        import seaborn as sns
        import matplotlib.pyplot as plt
        from src.data_preprocessing import create_feature_df
        feat_df, _, _, _ = create_feature_df(os.path.join(os.path.dirname(__file__), 'data', 'premier_league_2024_2025.csv'))
        corr = feat_df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=False)
        ax.set_title("Correlation Heatmap of Features")
        st.pyplot(fig)

    # Feature importance tab
    with tabs[2]:
        st.header("Feature Importance")
        st.write("Understanding which features drive the model’s predictions can be insightful.")
        try:
            import shap
            # Compute SHAP values on a subset to speed up rendering
            X, y, feature_names, _ = create_feature_df(os.path.join(os.path.dirname(__file__), 'data', 'premier_league_2024_2025.csv'))
            X_scaled = scaler.transform(X)
            explainer = shap.Explainer(model)
            shap_values = explainer(X_scaled)
            # Summarize feature importance
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            # Fallback: plot feature importances from tree-based models or coefficients
            st.warning(f"SHAP values could not be computed. Showing alternate feature importance. ({e})")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                ax.bar(range(len(importances)), importances[indices])
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                ax.set_title("Feature Importances")
                st.pyplot(fig)
            elif hasattr(model, 'coef_'):
                # Logistic regression coefficients
                coefs = model.coef_[0]
                indices = np.argsort(np.abs(coefs))[::-1]
                ax.bar(range(len(coefs)), coefs[indices])
                ax.set_xticks(range(len(coefs)))
                ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                ax.set_title("Feature Coefficients")
                st.pyplot(fig)
            else:
                st.write("No feature importance available for this model type.")


if __name__ == '__main__':
    main()