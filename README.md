# NeuroFive ML Track

This repository contains my work for the NeuroFive ML Track internship. The first three tasks use the Titanic dataset to progress from exploratory data analysis and data cleaning to a machine learning classification model.

## Tasks Completed

### Task 1 — Exploratory Data Analysis (EDA)

**Notebook:** `Titanic_EDA.ipynb`

The first task focused on understanding the Titanic dataset before applying machine learning.

#### Work Completed
- Loaded the Titanic training dataset using Pandas.
- Inspected the dataset using `head()`, `info()`, and `describe()`.
- Identified the dataset shape and feature types.
- Checked for missing values.
- Identified numerical and categorical features.
- Added a short data story summarizing the initial findings.

#### Dataset
- 891 passenger records
- 12 features
- Target variable: `Survived`

The dataset contains both numerical and categorical features, with missing values mainly present in `Cabin`, `Age`, and `Embarked`.

---

### Task 2 — Data Cleaning & Visualization

**Notebook:** `Task_02_Titanic_Data_Cleaning_Visualization.ipynb`

The second task focused on cleaning the Titanic dataset and using visualization to identify patterns and outliers.

#### Data Cleaning
- Missing `Age` values were filled using the median.
- Missing `Embarked` values were filled using the most frequent value.
- The `Cabin` column was removed because a large proportion of its values were missing.

#### Visualizations
The notebook includes:
- Age distribution histogram
- Fare boxplot for outlier detection
- Survival rate bar chart by gender
- Correlation heatmap

#### Key Finding
Based on the visualizations and exploratory analysis, `Sex` appears to have the strongest relationship with survival. Female passengers had a significantly higher survival rate than male passengers.

---

### Task 3 — Titanic Survival Classification

**Notebook:** `Task_03_Titanic_Classification.ipynb`

The third task used the cleaned Titanic dataset to build a classification model for predicting passenger survival.

#### Approach

1. Loaded the Titanic dataset.
2. Handled missing values in `Age` and `Embarked`.
3. Removed the `Cabin` column.
4. Selected relevant features:
   - `Pclass`
   - `Sex`
   - `Age`
   - `SibSp`
   - `Parch`
   - `Fare`
   - `Embarked`
5. Encoded categorical features (`Sex` and `Embarked`) using `OneHotEncoder`.
6. Split the dataset into training and testing sets using an 80/20 split.
7. Trained a Logistic Regression classification model.
8. Evaluated the model using accuracy and a confusion matrix.

#### Model

**Logistic Regression**

The model was implemented using a Scikit-learn pipeline containing a `ColumnTransformer`, `OneHotEncoder`, and Logistic Regression classifier.

#### Train/Test Split

- Training samples: 712
- Testing samples: 179

#### Final Accuracy

**80.45%**

The model achieved an accuracy of **0.8045 (80.45%)** on the test set.

#### Confusion Matrix

```text
[[98 12]
 [23 46]]
