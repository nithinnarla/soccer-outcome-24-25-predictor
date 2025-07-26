"""
Script for training a soccer match outcome prediction model on the Premier
League 2024/25 season data.  This module performs the following steps:

1. Loads the raw match results CSV from the project's `data` folder.
2. Generates engineered features using the Elo rating system and recent form
   statistics via ``create_feature_df`` from ``data_preprocessing``.
3. Splits the dataset into training and test sets (80/20 split).
4. Scales the features using ``StandardScaler`` to normalize numerical
   variables.
5. Evaluates multiple models using 5-fold cross validation with macro F1 score
   as the performance metric.  Currently tested models include:

   * Logistic Regression
   * Random Forest
   * XGBoost (if the xgboost package is installed)

6. Selects the best-performing model, retrains it on the full training set and
   evaluates it on the held-out test set, printing classification metrics.
7. Saves the trained model, scaler, feature names and label map to the
   ``models`` directory for later use by the Streamlit app.

Usage::

    python -m src.train_model

The script assumes it is executed from the project root (so that relative
paths resolve correctly).  You can also call the ``train`` function
programmatically to obtain the model objects without saving them.
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    has_xgboost = False

from .data_preprocessing import create_feature_df


def train(random_state: int = 42) -> Tuple[Any, StandardScaler, list, Dict[int, str]]:
    """Train models on the engineered dataset and return the best model.

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        A tuple ``(best_model, scaler, feature_names, label_map)``.
    """
    # Locate the raw data file within the project structure
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             'data', 'premier_league_2024_2025.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    X, y, feature_names, label_map = create_feature_df(data_path)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define candidate models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state)
    }
    if has_xgboost:
        models['XGBoost'] = XGBClassifier(
            objective='multi:softprob',
            num_class=len(np.unique(y)),
            eval_metric='mlogloss',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=random_state
        )

    # Evaluate models via cross-validation
    cv_scores = {}
    for name, model in models.items():
        try:
            # XGBoost accepts numpy arrays
            scores = cross_val_score(model, X_train_scaled, y_train, cv=5,
                                     scoring='f1_macro')
            cv_scores[name] = scores.mean()
            print(f"CV F1 score for {name}: {scores.mean():.4f} ± {scores.std():.4f}")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    # Select the best model
    best_name = max(cv_scores, key=cv_scores.get)
    best_model = models[best_name]
    print(f"Selected best model: {best_name}")

    # Train best model on full training set
    best_model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Test F1 (macro):", f1_score(y_test, y_pred, average='macro'))
    print("Classification report:\n", classification_report(y_test, y_pred))

    return best_model, scaler, feature_names, label_map


def main() -> None:
    """Entry point for the training script.  Trains and saves the model."""
    best_model, scaler, feature_names, label_map = train()
    # Save the model and ancillary objects
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_bundle = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'label_map': label_map
    }
    out_path = os.path.join(model_dir, 'soccer_outcome_model.pkl')
    joblib.dump(model_bundle, out_path)
    print(f"Model saved to {out_path}")


if __name__ == '__main__':
    main()