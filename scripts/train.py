"""
train.py
--------
Trains the best model using Konstantin's 6-feature pipeline.

Usage:
    python scripts/train.py

Output:
    scripts/model.pkl
    scripts/scaler.pkl
"""

import os, sys, pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import build_features

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
TEST_PATH  = os.path.join(BASE_DIR, 'data', 'test.csv')
MODEL_OUT  = os.path.join(BASE_DIR, 'scripts', 'model.pkl')
SCALER_OUT = os.path.join(BASE_DIR, 'scripts', 'scaler.pkl')


def main():
    print('Loading data...')
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    print('Engineering features...')
    X_train, y_train, _ = build_features(train_df, test_df)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # KNN with n=6 — Konstantin's best configuration (0.83253)
    model  = KNeighborsClassifier(algorithm='auto', leaf_size=26,
                                  n_neighbors=6, weights='uniform')
    kfold  = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_scaled, y_train, cv=kfold, scoring='accuracy')
    print(f'10-Fold CV: {scores.mean():.4f} ± {scores.std():.4f}')

    model.fit(X_scaled, y_train)
    print(f'Train accuracy: {model.score(X_scaled, y_train):.4f}')

    with open(MODEL_OUT,  'wb') as f: pickle.dump(model,  f)
    with open(SCALER_OUT, 'wb') as f: pickle.dump(scaler, f)
    print(f'Saved: {MODEL_OUT} | {SCALER_OUT}')


if __name__ == '__main__':
    main()
