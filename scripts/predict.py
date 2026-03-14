"""
predict.py
----------
Generates submission.csv using the trained model.

Usage:
    python scripts/predict.py

Output:
    data/submission.csv
"""

import os, sys, pickle
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import build_features

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH  = os.path.join(BASE_DIR, 'data', 'train.csv')
TEST_PATH   = os.path.join(BASE_DIR, 'data', 'test.csv')
MODEL_PATH  = os.path.join(BASE_DIR, 'scripts', 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scripts', 'scaler.pkl')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'submission.csv')


def main():
    with open(MODEL_PATH,  'rb') as f: model  = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    _, _, X_test = build_features(train_df, test_df)
    X_test_sc    = scaler.transform(X_test)
    preds        = model.predict(X_test_sc)

    pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived':    preds.astype(int)
    }).to_csv(OUTPUT_PATH, index=False)

    print(f'Saved: {OUTPUT_PATH}')
    print(f'Survivors: {preds.sum()}/{len(preds)} ({preds.mean():.1%})')


if __name__ == '__main__':
    main()
