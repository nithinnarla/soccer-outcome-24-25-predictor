"""
This module contains functions for reading the raw Premier League dataset and
creating a feature-rich dataframe for modeling.  The engineered features are
designed to capture both team strength (via an Elo rating system) and recent
form.  The input to these functions is a CSV file of matches with columns
`Date`, `Home Team`, `Away Team` and `Result` (e.g. "2 - 1").  The output
includes numerical features along with an encoded target label describing the
match outcome from the home team’s perspective.

Features engineered in this module include:

* **elo_diff** – difference between the home and away team Elo ratings prior to
  the match.  Elo ratings are initialized at 1500 for every team and updated
  after each game using a K-factor of 30.
* **home_form_points_avg / away_form_points_avg** – average points earned in
  each of the last five games for the home and away teams.  Points are
  calculated as 3 for a win, 1 for a draw and 0 for a loss.  At the start of
  the season (when fewer than five previous games exist) the feature is
  imputed with the mean of observed values later in the season.
* **home_goals_avg / away_goals_avg** – average number of goals scored over
  each team’s previous five matches.
* **home_conceded_avg / away_conceded_avg** – average number of goals
  conceded over each team’s previous five matches.
* **round_number** – the round number in which the match was played.
* **rest_days_home / rest_days_away** – number of days since each team’s
  previous match.  Missing values for rest days (at the start of the season)
  are filled with the median rest period.

The target column is returned as `target`, encoded as 0 for an away win,
1 for a draw and 2 for a home win.  A mapping dictionary is also returned
for reference.

Example usage::

    from src.data_preprocessing import create_feature_df
    X, y, feature_names, label_map = create_feature_df('data/premier_league_2024_2025.csv')

This module is intentionally self-contained so that it can be imported by the
model training script and the Streamlit app without re-computing features each
time.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict


def parse_score(score_str: str) -> Tuple[int, int]:
    """Parse the ``Result`` string into home and away goals.

    The result column in the raw dataset uses the format ``"x - y"``.  This
    helper splits the string on the hyphen and returns integers for home and
    away goals.  If the string is malformed, a ValueError is raised.

    Args:
        score_str: A score in the form "x - y".

    Returns:
        A tuple ``(home_goals, away_goals)``.
    """
    if not isinstance(score_str, str) or '-' not in score_str:
        raise ValueError(f"Invalid score format: {score_str}")
    parts = score_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid score format: {score_str}")
    try:
        home_goals = int(parts[0].strip())
        away_goals = int(parts[1].strip())
    except Exception:
        raise ValueError(f"Invalid numeric values in score: {score_str}")
    return home_goals, away_goals


def update_elo_rating(rating_home: float, rating_away: float, result: float, k: float = 30.0) -> Tuple[float, float]:
    """Update Elo ratings for two teams after a match.

    Args:
        rating_home: Current Elo rating for the home team.
        rating_away: Current Elo rating for the away team.
        result: Match outcome from the home team’s perspective: 1 for win,
            0.5 for draw, 0 for loss.
        k: K-factor controlling adjustment magnitude.  Default is 30.

    Returns:
        A tuple ``(new_home_rating, new_away_rating)``.
    """
    # Expected score for the home team
    expected_home = 1.0 / (1.0 + 10 ** (-(rating_home - rating_away) / 400))
    expected_away = 1.0 - expected_home
    # Update ratings
    new_home = rating_home + k * (result - expected_home)
    new_away = rating_away + k * ((1 - result) - expected_away)
    return new_home, new_away


def create_feature_df(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[int, str]]:
    """Create a dataframe of engineered features and target labels from the raw CSV.

    Args:
        csv_path: Path to the raw CSV file containing match results.

    Returns:
        A tuple ``(X, y, feature_names, label_map)`` where ``X`` is a
        feature dataframe, ``y`` is a Series of target labels, ``feature_names``
        is a list of feature column names and ``label_map`` maps the encoded
        target integers back to result strings.
    """
    df = pd.read_csv(csv_path)
    # Ensure proper column names regardless of case variations
    df.columns = [c.strip() for c in df.columns]
    required_cols = {"Date", "Home Team", "Away Team", "Result", "Round Number"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Parse date into datetime
    # The format includes time and uses day/month/year.  We coerce errors to NaT.
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M", errors='coerce')
    if df['Date'].isna().any():
        raise ValueError("Some dates could not be parsed; check the date format.")

    # Parse score into home and away goals
    home_goals_list = []
    away_goals_list = []
    for score in df['Result']:
        h, a = parse_score(score)
        home_goals_list.append(h)
        away_goals_list.append(a)
    df['home_goals'] = home_goals_list
    df['away_goals'] = away_goals_list

    # Determine match result from home team perspective: 1=home win, 0.5=draw, 0=away win
    def outcome_label(h: int, a: int) -> float:
        if h > a:
            return 1.0
        elif h == a:
            return 0.5
        else:
            return 0.0

    df['home_outcome'] = [outcome_label(h, a) for h, a in zip(df['home_goals'], df['away_goals'])]

    # Initialize dictionaries for elo ratings, recent results, goals and last played date
    teams = pd.concat([df['Home Team'], df['Away Team']]).unique()
    elo_ratings: Dict[str, float] = {team: 1500.0 for team in teams}
    results_history: Dict[str, List[float]] = {team: [] for team in teams}
    goals_for_history: Dict[str, List[int]] = {team: [] for team in teams}
    goals_against_history: Dict[str, List[int]] = {team: [] for team in teams}
    last_played: Dict[str, datetime] = {team: None for team in teams}

    # Containers for features
    elo_diffs = []
    home_form_points = []
    away_form_points = []
    home_goals_avg = []
    away_goals_avg = []
    home_conceded_avg = []
    away_conceded_avg = []
    rest_days_home = []
    rest_days_away = []

    for idx, row in df.sort_values('Date').iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']
        match_date: datetime = row['Date']

        # Current Elo difference
        elo_diff = elo_ratings[home_team] - elo_ratings[away_team]
        elo_diffs.append(elo_diff)

        # Form (average points in last 5 games) – convert outcomes into points (3/1/0)
        def calc_points(history: List[float]) -> float:
            if not history:
                return np.nan
            # Convert 1,0.5,0 to points 3,1,0
            pts = [3 * r if r in (1.0, 0.5, 0.0) else 0 for r in history]
            # Consider last five matches
            last5 = pts[-5:]
            return np.mean(last5)

        home_points_avg = calc_points(results_history[home_team])
        away_points_avg = calc_points(results_history[away_team])
        home_form_points.append(home_points_avg)
        away_form_points.append(away_points_avg)

        # Goals scored/conceded average over last 5 matches
        def calc_avg(lst: List[int]) -> float:
            if not lst:
                return np.nan
            return np.mean(lst[-5:])

        home_goals_avg.append(calc_avg(goals_for_history[home_team]))
        away_goals_avg.append(calc_avg(goals_for_history[away_team]))
        home_conceded_avg.append(calc_avg(goals_against_history[home_team]))
        away_conceded_avg.append(calc_avg(goals_against_history[away_team]))

        # Rest days (difference in days since last match)
        def calc_rest_days(team: str) -> float:
            last_date = last_played[team]
            if last_date is None:
                return np.nan
            return (match_date - last_date).days

        rest_days_home.append(calc_rest_days(home_team))
        rest_days_away.append(calc_rest_days(away_team))

        # Update histories after computing features
        # Append current outcome and goals
        result = row['home_outcome']  # 1.0, 0.5, or 0.0
        # Append to result history for both teams (home perspective for home, 1-result for away)
        results_history[home_team].append(result)
        # For away team, invert: if home wins (1.0), away loses (0.0); draw stays 0.5; home loses (0.0) => away wins (1.0)
        results_history[away_team].append(1.0 - result)

        # Append goals for/against
        h_goals = row['home_goals']
        a_goals = row['away_goals']
        goals_for_history[home_team].append(h_goals)
        goals_against_history[home_team].append(a_goals)
        goals_for_history[away_team].append(a_goals)
        goals_against_history[away_team].append(h_goals)

        # Update last played date
        last_played[home_team] = match_date
        last_played[away_team] = match_date

        # Update Elo ratings after the match
        new_home_rating, new_away_rating = update_elo_rating(
            rating_home=elo_ratings[home_team],
            rating_away=elo_ratings[away_team],
            result=result,
            k=30.0
        )
        elo_ratings[home_team] = new_home_rating
        elo_ratings[away_team] = new_away_rating

    # Assign feature columns
    features = pd.DataFrame({
        'elo_diff': elo_diffs,
        'home_form_points_avg': home_form_points,
        'away_form_points_avg': away_form_points,
        'home_goals_avg': home_goals_avg,
        'away_goals_avg': away_goals_avg,
        'home_conceded_avg': home_conceded_avg,
        'away_conceded_avg': away_conceded_avg,
        'rest_days_home': rest_days_home,
        'rest_days_away': rest_days_away,
        'round_number': df['Round Number'].astype(float)
    })

    # Replace missing values with column medians
    features = features.fillna(features.median())

    # Create target label mapping: 0 => away win, 1 => draw, 2 => home win
    label_map = {0: 'A', 1: 'D', 2: 'H'}
    def encode_target(outcome: float) -> int:
        # outcome is 1.0 (home win), 0.5 (draw), 0.0 (away win)
        if outcome == 1.0:
            return 2
        elif outcome == 0.5:
            return 1
        else:
            return 0
    target = df['home_outcome'].apply(encode_target)

    feature_names = list(features.columns)
    return features, target, feature_names, label_map