"""
preprocessing.py
----------------
Reusable feature engineering functions for the Titanic dataset.

Functions:
    extract_group_survival(df)  — adds Family_Survival column
    clean_data(df)              — adds Family_Size, Last_Name
"""

import pandas as pd
import numpy as np


def extract_group_survival(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Family_Survival column to the combined train+test dataframe.

    Logic:
    - Group passengers by Last_Name + Fare (same family = same surname + same ticket price)
    - If any family member's survival is known → propagate it
    - Then refine using Ticket number (travel companions who aren't family)
    - Default value = mean survival rate (0.5 fallback if no info available)

    Why this feature matters:
    Passengers travelling together had correlated survival outcomes —
    they helped each other reach lifeboats. This is the single highest-impact
    engineered feature for pushing scores above 80%.

    Args:
        df: combined train + test dataframe (must contain Survived, Name, Fare, Ticket)

    Returns:
        df with added columns: Last_Name, Family_Survival
    """
    df = df.copy()

    # Extract last name for family grouping
    df['Last_Name'] = df['Name'].apply(lambda x: x.split(',')[0].strip())

    # Fill missing Fare so groupby works cleanly
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Default: overall mean survival rate (used when no family info is available)
    MEAN_SURVIVAL = round(df['Survived'].mean(), 4)
    df['Family_Survival'] = MEAN_SURVIVAL

    # Pass 1: group by last name + fare (same family travelling together)
    for _, grp in df.groupby(['Last_Name', 'Fare']):
        if len(grp) > 1:
            for idx, row in grp.iterrows():
                others = grp.drop(idx)
                smax = others['Survived'].max()
                smin = others['Survived'].min()
                pid  = row['PassengerId']
                if smax == 1.0:
                    df.loc[df['PassengerId'] == pid, 'Family_Survival'] = 1.0
                elif smin == 0.0:
                    df.loc[df['PassengerId'] == pid, 'Family_Survival'] = 0.0

    # Pass 2: refine using ticket number (non-family travel companions)
    for _, grp in df.groupby('Ticket'):
        if len(grp) > 1:
            for idx, row in grp.iterrows():
                if row['Family_Survival'] in [0.0, MEAN_SURVIVAL]:
                    others = grp.drop(idx)
                    smax = others['Survived'].max()
                    smin = others['Survived'].min()
                    pid  = row['PassengerId']
                    if smax == 1.0:
                        df.loc[df['PassengerId'] == pid, 'Family_Survival'] = 1.0
                    elif smin == 0.0:
                        df.loc[df['PassengerId'] == pid, 'Family_Survival'] = 0.0

    n = (df['Family_Survival'] != MEAN_SURVIVAL).sum()
    print(f"  [extract_group_survival] Passengers with family/group info: {n} / {len(df)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Family_Size feature to the dataframe.

    Family_Size = SibSp + Parch + 1 (the passenger themselves).
    This combines two separate family columns into one total group size.

    Args:
        df: dataframe with SibSp and Parch columns

    Returns:
        df with added Family_Size column
    """
    df = df.copy()
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    print(f"  [clean_data] Family_Size created. Range: {df['Family_Size'].min()}–{df['Family_Size'].max()}")
    return df
