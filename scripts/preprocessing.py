"""
preprocessing.py
----------------
Feature pipeline based on Konstantin Masich's 0.82-0.83 Kaggle kernel.
https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83

Exact 6 features used:
    Pclass, Sex, Family_Size, Family_Survival, FareBin_Code, AgeBin_Code

Key decisions from the kernel:
- Family_Size = SibSp + Parch (no +1)
- Embarked is dropped — confirmed to have no impact on survival
- FareBin and AgeBin use label-encoded quantile bins (ordinal, not one-hot)
  because FareBin=3 IS greater than FareBin=1
- Family_Survival: 0, 0.5 (default), or 1
  computed via Last_Name+Fare grouping first, then Ticket grouping
- StandardScaler applied before KNN
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def _impute_age(data_df, train_df, test_df):
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

    mapping = {
        'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',
        'Don': 'Mr', 'Mme': 'Miss', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
        'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'
    }
    data_df.replace({'Title': mapping}, inplace=True)

    titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
    for title in titles:
        median_age = data_df.groupby('Title')['Age'].median()[title]
        mask = (data_df['Age'].isnull()) & (data_df['Title'] == title)
        data_df.loc[mask, 'Age'] = median_age

    train_df['Age'] = data_df['Age'][:len(train_df)].values
    test_df['Age']  = data_df['Age'][len(train_df):].values

    data_df.drop('Title', axis=1, inplace=True)

    return data_df, train_df, test_df


def _family_survival(data_df, train_df, test_df):
    """
    Compute Family_Survival exactly as in Konstantin's kernel.

    Values: 0 (group died), 0.5 (unknown/default), 1 (group survived)

    Pass 1: group by Last_Name + Fare (same family)
    Pass 2: group by Ticket (travel companions)
    """
    data_df['Last_Name'] = data_df['Name'].apply(lambda x: str.split(x, ',')[0])
    data_df['Fare'].fillna(data_df['Fare'].mean(), inplace=True)

    DEFAULT = 0.5
    data_df['Family_Survival'] = DEFAULT

    # Pass 1: families
    for _, grp_df in data_df[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket',
                               'PassengerId', 'SibSp', 'Parch', 'Age', 'Cabin']]\
                        .groupby(['Last_Name', 'Fare']):
        if len(grp_df) != 1:
            for ind, row in grp_df.iterrows():
                smax   = grp_df.drop(ind)['Survived'].max()
                smin   = grp_df.drop(ind)['Survived'].min()
                pid    = row['PassengerId']
                if smax == 1.0:
                    data_df.loc[data_df['PassengerId'] == pid, 'Family_Survival'] = 1
                elif smin == 0.0:
                    data_df.loc[data_df['PassengerId'] == pid, 'Family_Survival'] = 0

    print(f"  Pass 1 — passengers with family info: "
          f"{data_df.loc[data_df['Family_Survival'] != DEFAULT].shape[0]}")

    # Pass 2: ticket companions
    for _, grp_df in data_df.groupby('Ticket'):
        if len(grp_df) != 1:
            for ind, row in grp_df.iterrows():
                if (row['Family_Survival'] == 0) | (row['Family_Survival'] == DEFAULT):
                    smax = grp_df.drop(ind)['Survived'].max()
                    smin = grp_df.drop(ind)['Survived'].min()
                    pid  = row['PassengerId']
                    if smax == 1.0:
                        data_df.loc[data_df['PassengerId'] == pid, 'Family_Survival'] = 1
                    elif smin == 0.0:
                        data_df.loc[data_df['PassengerId'] == pid, 'Family_Survival'] = 0

    print(f"  Pass 2 — passengers with family/group info: "
          f"{data_df.loc[data_df['Family_Survival'] != DEFAULT].shape[0]}")

    # Push back to train/test
    train_df['Family_Survival'] = data_df['Family_Survival'][:len(train_df)].values
    test_df['Family_Survival']  = data_df['Family_Survival'][len(train_df):].values

    return data_df, train_df, test_df


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Exact feature pipeline from Konstantin's 0.82-0.83 kernel.

    Final features (6 total):
        Pclass, Sex, Family_Size, Family_Survival, FareBin_Code, AgeBin_Code

    Returns: X_train, y_train, X_test
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    # Combined dataset for consistent transformations
    data_df = pd.concat([train_df, test_df], ignore_index=True)

    # --- Age imputation via Title (Title dropped after) ---
    data_df, train_df, test_df = _impute_age(data_df, train_df, test_df)

    # --- Family Size: SibSp + Parch (no +1, matches kernel exactly) ---
    data_df['Family_Size'] = data_df['Parch'] + data_df['SibSp']
    train_df['Family_Size'] = data_df['Family_Size'][:len(train_df)].values
    test_df['Family_Size']  = data_df['Family_Size'][len(train_df):].values

    # --- Family Survival ---
    data_df, train_df, test_df = _family_survival(data_df, train_df, test_df)

    # --- Fare Bins (quantile, label encoded — ordinal feature) ---
    data_df['Fare'].fillna(data_df['Fare'].median(), inplace=True)
    data_df['FareBin'] = pd.qcut(data_df['Fare'], 5)
    label = LabelEncoder()
    data_df['FareBin_Code'] = label.fit_transform(data_df['FareBin'])
    train_df['FareBin_Code'] = data_df['FareBin_Code'][:len(train_df)].values
    test_df['FareBin_Code']  = data_df['FareBin_Code'][len(train_df):].values

    # --- Age Bins (quantile, label encoded — ordinal feature) ---
    data_df['AgeBin'] = pd.qcut(data_df['Age'], 4)
    data_df['AgeBin_Code'] = label.fit_transform(data_df['AgeBin'])
    train_df['AgeBin_Code'] = data_df['AgeBin_Code'][:len(train_df)].values
    test_df['AgeBin_Code']  = data_df['AgeBin_Code'][len(train_df):].values

    # --- Sex encoding: male=0, female=1 ---
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex']  = test_df['Sex'].map({'male': 0, 'female': 1})

    # --- Final feature selection (exactly Konstantin's 6 features) ---
    FEATURES = ['Pclass', 'Sex', 'Family_Size', 'Family_Survival',
                'FareBin_Code', 'AgeBin_Code']

    X_train = train_df[FEATURES]
    y_train = train_df['Survived'].astype(int)
    X_test  = test_df[FEATURES]

    print(f"\n  Features ({len(FEATURES)}): {FEATURES}")
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  Missing in X_train: {X_train.isna().sum().sum()}")

    return X_train, y_train, X_test
