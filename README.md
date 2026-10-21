# Medical Insurance Charges: What Drives Cost?

## Problem / Research Question

This project analyzes a medical insurance dataset to answer the question:

**Which factors — particularly smoking status and BMI — are most strongly associated with individual medical insurance charges, and how well can a regression model explain and predict those charges?**

## Dataset Source and Usage Permissions

- **Source:** `medical_insurance_dataset.csv`, provided through the IBM Developer Skills Network DA0101EN course dataset hosted on IBM Cloud Object Storage.
- **Contents:** 7 columns — `age`, `gender`, `bmi`, `no_of_children`, `smoker`, `region`, `charges`.
- **Usage:** The dataset is a public, anonymized teaching dataset distributed for educational use. It contains no personally identifiable information (no names, IDs, addresses, or contact details). It is used here solely for coursework and portfolio demonstration.

## Methods and Tools

**Tools / Libraries**
- Python 3.13
- `pandas` — data loading and cleaning
- `numpy` — numeric operations
- `matplotlib` / `seaborn` — visualization
- `scikit-learn` — regression modeling and evaluation

**Data Cleaning**
- Loaded the CSV without a header row and assigned column names manually.
- Replaced `"?"` placeholders with `NaN`.
- Imputed missing `age` values with the column mean.
- Imputed missing `smoker` values with the most frequent category.
- Cast `age` and `smoker` to integer and rounded `charges` to 2 decimal places.

**Analysis and Modeling**
1. **Exploratory visualization:** a regression plot of `bmi` vs. `charges`, and a box plot of `charges` by `smoker` status.
2. **Simple linear regression:** `charges` ~ `smoker`.
3. **Multiple linear regression:** `charges` ~ all six predictors (`age`, `gender`, `bmi`, `no_of_children`, `smoker`, `region`).
4. **Polynomial pipeline:** `StandardScaler` → `PolynomialFeatures` → `LinearRegression`.
5. **Ridge regression:** trained on an 80/20 train–test split (`random_state=1`, `alpha=0.1`), both on raw features and on degree-2 polynomial features.

## Key Findings and One Limitation

**Key findings**
- **Smoking status is the dominant driver of charges.** The box plot shows smokers' charges are substantially higher and more dispersed than non-smokers', and the simple linear regression on `smoker` alone already captures a large share of the variation in `charges`.
- **BMI has a positive relationship with charges**, visible in the regression plot, but the spread is wide — BMI alone is a weak predictor.
- **Adding predictors improves fit.** The multiple regression and polynomial/Ridge models capture more structure than the single-variable model, indicating that charges depend on a combination of demographic and lifestyle factors rather than any one variable.

**Limitation**
- The models are evaluated on a single 80/20 split and the notebook does not report cross-validated R² or RMSE, so the performance figures are sensitive to the particular split and should not be read as out-of-sample predictive accuracy. In addition, the categorical variables (`gender`, `region`) are passed to the models as-is without explicit one-hot encoding, which limits how much of their effect the models can capture.

## Instructions for Running the Analysis

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/data-science-project.git
cd data-science-project
