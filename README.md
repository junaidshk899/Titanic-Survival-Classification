# 🚢 Titanic Survival Classification — Machine Learning Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Task 1 — Data Science Internship @ Arch Technologies**

*Predicting survival outcomes of Titanic passengers using supervised machine learning*

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Technologies Used](#-technologies-used)
- [Installation & Setup](#-installation--setup)
- [Implementation Details](#-implementation-details)
  - [Data Cleaning & Preprocessing](#1-data-cleaning--preprocessing)
  - [Feature Engineering](#2-feature-engineering)
  - [Model Training](#3-model-training)
  - [Model Evaluation](#4-model-evaluation)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🎯 Project Overview

This project is **Task 1** of my Data Science Internship at **Arch Technologies**. The goal is to build a binary classification model that predicts whether a Titanic passenger **survived (1)** or **did not survive (0)** based on features such as gender, age, ticket class, and fare.

The project covers a complete end-to-end machine learning pipeline — from raw data exploration and preprocessing, through feature engineering and model training, to final evaluation and interpretation.

> **Best Model:** Random Forest Classifier — **82.68% Test Accuracy** | **0.8812 AUC**

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [Kaggle — Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) |
| Rows | 891 passengers |
| Columns | 12 features |
| Target | `Survived` (0 = Did not survive, 1 = Survived) |
| Class Balance | 61.6% not survived / 38.4% survived |

### Feature Summary

| Feature | Type | Description | Action Taken |
|---|---|---|---|
| `PassengerId` | int | Unique row ID | Dropped |
| `Survived` | int | Target variable (0/1) | Kept |
| `Pclass` | int | Ticket class (1, 2, 3) | Kept |
| `Name` | object | Passenger full name | Dropped |
| `Sex` | object | Gender | Label Encoded |
| `Age` | float | Age in years (177 missing) | Median Imputed |
| `SibSp` | int | Siblings/spouses aboard | Kept |
| `Parch` | int | Parents/children aboard | Kept |
| `Ticket` | object | Ticket number | Dropped |
| `Fare` | float | Ticket price in £ | Kept |
| `Cabin` | object | Cabin number (687 missing) | Dropped |
| `Embarked` | object | Port of embarkation (2 missing) | Mode Imputed + OHE |

---

## 🔄 Project Workflow

```
Raw Data  →  EDA  →  Cleaning  →  Feature Engineering  →  Encoding & Scaling  →  Model Training  →  Evaluation  →  Insights
```

1. **Data Loading & Exploration** — Shape, dtypes, missing values, statistics
2. **Exploratory Data Analysis (EDA)** — Visualize survival patterns across features
3. **Data Cleaning** — Drop irrelevant columns, impute missing values
4. **Feature Engineering** — Create `FamilySize`, `IsAlone`, `AgeGroup`
5. **Encoding & Scaling** — Label encode `Sex`, one-hot encode `Embarked`, StandardScaler
6. **Model Training** — Train 6 classifiers with 5-Fold Stratified Cross-Validation
7. **Evaluation** — Accuracy, Confusion Matrix, Classification Report, ROC-AUC
8. **Feature Importance** — Identify the most predictive variables

---

## 🛠 Technologies Used

| Library | Version | Purpose |
|---|---|---|
| Python | 3.x | Core programming language |
| pandas | 2.x | Data manipulation |
| NumPy | 1.x | Numerical operations |
| Matplotlib | 3.x | Data visualization |
| Seaborn | 0.x | Statistical visualizations |
| Scikit-learn | 1.x | ML models & evaluation |
| Google Colab | — | Cloud development environment |

---

## ⚙️ Installation & Setup

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the notebook: `Titanic_Survival_Classification.ipynb`
3. Upload the dataset: `Titanic-Dataset.csv` via the Files panel
4. Click **Runtime → Run all**

### Option 2 — Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/titanic-survival-classification.git
cd titanic-survival-classification

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook Titanic_Survival_Classification.ipynb
```

### Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
jupyter>=1.0.0
```

---

## 🔬 Implementation Details

### 1. Data Cleaning & Preprocessing

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Titanic-Dataset.csv')

# Drop irrelevant columns
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Impute missing values
# Age: median is robust against outliers
df['Age'] = df['Age'].fillna(df['Age'].median())

# Embarked: fill with most frequent port ("S")
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical variables
# Label Encoding for binary Sex column
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

# One-Hot Encoding for Embarked (3 categories → 2 binary columns)
# drop_first=True avoids the dummy variable trap
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

### 2. Feature Engineering

```python
# FamilySize: total family unit size aboard
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# IsAlone: binary flag for solo travelers
df['IsAlone'] = ((df['SibSp'] + df['Parch']) == 0).astype(int)

# AgeGroup: bin continuous age into life-stage categories
# 0=Child, 1=Teen, 2=Young Adult, 3=Adult, 4=Senior
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=[0, 1, 2, 3, 4]
)
```

### 3. Model Training

```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Split features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# 80/20 stratified split (preserves class ratio in both splits)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize features — fit ONLY on training data to prevent data leakage
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Define all 6 models
models = {
    'Logistic Regression':  LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':        DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM':                  SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors':  KNeighborsClassifier(n_neighbors=5)
}

# 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv)
    model.fit(X_train_sc, y_train)
    test_acc = model.score(X_test_sc, y_test)
    print(f'{name:<25}  CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  |  Test: {test_acc:.4f}')
```

### 4. Model Evaluation

```python
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)

# Best model — Random Forest
best_model = models['Random Forest']
y_pred = best_model.predict(X_test_sc)
y_prob = best_model.predict_proba(X_test_sc)[:, 1]

# Classification Report
print(classification_report(y_test, y_pred,
      target_names=['Did Not Survive', 'Survived']))

# ROC-AUC Score
auc = roc_auc_score(y_test, y_prob)
print(f'AUC Score: {auc:.4f}')
```

---

## 📈 Results

### Model Performance Summary

| Model | CV Mean Accuracy | CV Std | Test Accuracy | AUC Score |
|---|---|---|---|---|
| Logistic Regression | 80.48% | ±2.10% | 80.45% | 0.8671 |
| Decision Tree | 80.48% | ±3.23% | 75.98% | 0.7998 |
| **Random Forest** | **80.20%** | **±1.93%** | **82.68%** | **0.8812** |
| Gradient Boosting | 82.44% | ±2.57% | 80.45% | 0.8854 |
| SVM | 82.58% | ±2.18% | 82.12% | 0.8783 |
| K-Nearest Neighbors | 78.37% | ±2.89% | 81.56% | 0.8550 |

> ✅ **Winner: Random Forest** — Highest test accuracy (82.68%) with strong AUC (0.8812) and low variance across cross-validation folds.

### Classification Report (Random Forest)

```
                  precision    recall  f1-score   support

Did Not Survive       0.85      0.88      0.86       111
       Survived       0.79      0.74      0.76        68

       accuracy                           0.83       179
      macro avg       0.82      0.81      0.81       179
   weighted avg       0.83      0.83      0.83       179
```

---

## 💡 Key Insights

1. **Gender is the strongest predictor** — Female passengers had a 74.2% survival rate vs. 18.9% for males, directly reflecting the "women and children first" evacuation protocol.

2. **Passenger class drove access to lifeboats** — 1st class passengers had a 62.9% survival rate vs. only 24.2% for 3rd class, reflecting proximity to lifeboat decks.

3. **Feature engineering added value** — `FamilySize` and `IsAlone` captured social dynamics not present in raw `SibSp`/`Parch` columns individually.

4. **Preprocessing matters more than model choice** — The gap between models was small (~7%), but the gap from proper preprocessing (imputation, encoding, scaling) to no preprocessing is much larger.

5. **Random Forest generalizes best** — Ensemble averaging over 100 trees reduced overfitting significantly compared to a single Decision Tree (82.68% vs 75.98% test accuracy).

---

## 📁 Project Structure

```
titanic-survival-classification/
│
├── 📓 Titanic_Survival_Classification.ipynb   # Main Google Colab notebook
├── 📄 Titanic-Dataset.csv                     # Raw dataset
├── 📊 Titanic_Task1_Report.pdf                # Full project report (PDF)
├── 📝 Titanic_Task1_Report.docx               # Full project report (Word)
├── 📋 requirements.txt                        # Python dependencies
└── 📖 README.md                               # This file
```

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/titanic-survival-classification.git

# Navigate into the project folder
cd titanic-survival-classification

# Install dependencies
pip install -r requirements.txt

# Open the notebook
jupyter notebook Titanic_Survival_Classification.ipynb
```

Or simply open in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/titanic-survival-classification/blob/main/Titanic_Survival_Classification.ipynb)

---

## 👤 Author

**Muhammad Junaid Asim**
Data Science Intern — Arch Technologies

[![Email](https://img.shields.io/badge/Email-Junaidasim899%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:Junaidasim899@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/YOUR_LINKEDIN_USERNAME)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/YOUR_USERNAME)

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

*Made with dedication during my Data Science Internship at Arch Technologies*

</div>
