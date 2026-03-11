# Kaggle Titanic — Machine Learning from Disaster

**Kaggle username:** `baidarbe01EDU_Astana_03_2026`

---

## Project Overview

This project tackles the classic Kaggle Titanic challenge: predicting which passengers survived the sinking of the RMS Titanic (April 15, 1912). The dataset contains passenger information — age, sex, ticket class, family size, and more.

**Goal:** Achieve **≥78.9% accuracy** on the Kaggle leaderboard through smart feature engineering and an ensemble model.

---

## Best Leaderboard Score

| Score | Notes |
|-------|-------|
| TBD   | Will be updated after submission |

**Local CV score (10-fold Stratified):** TBD

---

## Feature Engineering

The key insight: *put yourself in an investigator's shoes*. Raw columns tell one story; engineered features tell the real story.

| Feature | Source | Why It Helps |
|---|---|---|
| `Title_Code` | Regex from `Name` | Encodes gender + age group + social status (Mr, Mrs, Miss, Master, Rare) |
| `Family_Survival` | Group by LastName+Fare, then Ticket | Families evacuated together — biggest single boost |
| `FamilySize` | SibSp + Parch + 1 | Family of 2–4 survived best; solo and large groups worse |
| `IsAlone` | FamilySize == 1 | Binary flag: solo travelers had ~30% survival |
| `FamilySizeGroup` | 3-way bucket of FamilySize | Captures the non-linear family effect explicitly |
| `AgeBin` | qcut(Age, 4) | Non-linear age effect: children < 10 had priority |
| `IsChild` / `IsSenior` | Age thresholds | EDA showed elevated survival for children, lower for elderly |
| `FareBin` | qcut(Fare, 5) | Wealth proxy via quantile bins |
| `Pclass_Sex` | Pclass × Sex | Interaction: 1st-class female survived most (~97%) |
| `Pclass_Title` | Pclass × Title_Code | Richer encoding of class + social standing |
| `Ticket_Frequency` | Group by Ticket | Number of people sharing a ticket = travel group size |

**Features that do NOT help** (per top Kaggle kernels):
- `Has_Cabin` binary flag — noisy, inconsistent with survival
- `Deck` from cabin letter — 77% unknown, too sparse
- `Embarked` alone — mostly a proxy for Pclass, adds noise

---

## Project Structure

```
kaggle-titanic/
│   README.md
│   environment.yml
│   requirements.txt
│   username.txt
│   .gitignore
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── submission.csv          ← generated after running predict.py
│
├── notebook/
│   └── EDA.ipynb               ← Full pipeline: EDA → Feature Engineering → Model
│
└── scripts/
    ├── train.py                ← Train best model, save model.pkl + scaler.pkl
    └── predict.py              ← Load model, generate submission.csv
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/eloquenttbota/kaggle-titanic.git 
```
or 
```
git clone https://01.tomorrow-school.ai/git/baidarbe/kaggle-titanic.git
```
``` 
cd kaggle-titanic
```

### 2. Create virtual environment

```bash
python -m venv tomorrow
```
or
```
python3 -m venv tomorrow
```
```
source tomorrow/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```


### 4. Run the full notebook (EDA + model)

```bash
jupyter notebook notebook/EDA.ipynb
```

Run all cells top-to-bottom. Final cell saves `data/submission.csv`.

### 5. Or use the scripts directly

```bash
# Train the model (saves model.pkl + scaler.pkl to scripts/)
python scripts/train.py

# Generate predictions (saves data/submission.csv)
python scripts/predict.py
```

---

## Resources

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Top-2% Solution: Ultimate EDA & Feature Engineering](https://www.kaggle.com/sreevishnudamodaran/ultimate-eda-fe-neural-network-model-top-2)
- [KNN-based 0.82-0.83 Solution](https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83)
- [Scikit-learn Documentation](https://scikit-learn.org/)