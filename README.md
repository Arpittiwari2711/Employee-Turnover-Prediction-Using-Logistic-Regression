# 👥 Employee Turnover Prediction Using Logistic Regression

## 📌 Project Overview

**TalentCore Pvt. Ltd.** is a multinational company experiencing an increase in employee resignations. High employee turnover leads to increased recruitment costs, project delays, loss of skilled employees, and additional pressure on the HR department.

This project develops a **Supervised Machine Learning Classification System** to predict whether an employee is likely to leave the company based on various work-related factors.

The project uses Logistic Regression to classify employees into two categories:

* **Employee Stayed:** The employee continues working for the company.
* **Employee Left:** The employee leaves the company.

Three Logistic Regression models are implemented and compared:

1. Baseline Logistic Regression
2. Logistic Regression with L1 Regularization
3. Logistic Regression with L2 Regularization

The objective is to evaluate the performance of these models and identify the most suitable model for employee turnover prediction.

---

# 🎯 Problem Statement

TalentCore Pvt. Ltd. wants to develop an intelligent Machine Learning system that predicts employee turnover using employee-related information, including job satisfaction, salary, age, work-life balance, training hours, annual bonuses, and other workplace factors.

As an AI/ML Engineer, the primary objectives are:

* Build a baseline Logistic Regression model.
* Improve the model using L1 Regularization.
* Implement L2 Regularization to control model complexity.
* Evaluate and compare the performance of all three models.
* Recommend the best-performing model based on evaluation results.

The target variable is:

```text
Employee_Turnover
```

Target variable interpretation:

| Value | Meaning         |
| ----- | --------------- |
| `0`   | Employee Stayed |
| `1`   | Employee Left   |

This is a **Binary Classification Problem** because the model predicts one of two possible outcomes.

---

# 🤖 Machine Learning Approach

This project follows the following Machine Learning approach:

```text
Supervised Learning
        ↓
Classification
        ↓
Binary Classification
        ↓
Logistic Regression
        ↓
Regularization
   ↙     ↓     ↘
Baseline  L1    L2
   ↘     ↓     ↙
Model Performance Comparison
        ↓
Best Model Recommendation
```

## 🔄 Prediction Workflow

The project follows this workflow:

```text
Employee Dataset
        ↓
Data Loading
        ↓
Feature and Target Separation
        ↓
Train-Test Split
        ↓
Baseline Logistic Regression
        ↓
L1 Regularization
        ↓
L2 Regularization
        ↓
Model Training
        ↓
Employee Turnover Prediction
        ↓
Model Evaluation
        ↓
Performance Comparison
        ↓
Best Model Selection
```

---

# 📊 Dataset

The project uses the following dataset:

**Dataset Name:** `employee_turnover.csv`

**Dataset Size:** 1,350 rows and 16 columns.

The dataset contains employee-related metrics used to predict whether an employee leaves or stays with the company.

It includes 15 input features and one target variable.

> Note: The original assignment description mentions 900 rows and 15 columns. However, the supplied CSV file contains 1,350 rows and 16 columns, including 15 input features and the target variable. This README documents the supplied dataset.

## 📋 Dataset Features

| Feature                                   | Description                                                    |
| ----------------------------------------- | -------------------------------------------------------------- |
| `Job_Satisfaction`                        | Employee's level of satisfaction with the job                  |
| `Performance_Rating`                      | Employee performance score                                     |
| `Years_At_Company`                        | Number of years the employee has worked at the company         |
| `Work_Life_Balance`                       | Measure of balance between professional and personal life      |
| `Distance_From_Home`                      | Distance between the employee's home and workplace             |
| `Monthly_Income`                          | Employee's monthly salary                                      |
| `Education_Level`                         | Employee's education qualification level                       |
| `Age`                                     | Age of the employee                                            |
| `Num_Companies_Worked`                    | Number of companies the employee has previously worked for     |
| `Employee_Role`                           | Encoded employee job role                                      |
| `Annual_Bonus`                            | Annual bonus received by the employee                          |
| `Training_Hours`                          | Number of training hours attended                              |
| `Department`                              | Encoded department                                             |
| `Annual_Bonus_Squared`                    | Engineered feature representing the squared annual bonus       |
| `Annual_Bonus_Training_Hours_Interaction` | Interaction feature between annual bonus and training hours    |
| `Employee_Turnover`                       | Target variable indicating whether the employee left or stayed |

---

# 🎯 Features and Target Variable

## Input Features

The model uses the following 15 input features:

```text
Job_Satisfaction
Performance_Rating
Years_At_Company
Work_Life_Balance
Distance_From_Home
Monthly_Income
Education_Level
Age
Num_Companies_Worked
Employee_Role
Annual_Bonus
Training_Hours
Department
Annual_Bonus_Squared
Annual_Bonus_Training_Hours_Interaction
```

## Target Variable

```text
Employee_Turnover
```

The target variable represents the employee's turnover status.

* `0` indicates that the employee stayed with the company.
* `1` indicates that the employee left the company.

The target variable is separated from the input features before training the models.

```python
X = df.drop('Employee_Turnover', axis=1)
y = df['Employee_Turnover']
```

Here:

* `X` contains the input features.
* `y` contains the target variable.

---

# 🧹 Data Preparation

The dataset is loaded into a Pandas DataFrame before being used for model training.

## 1. Data Loading

The CSV file is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("employee_turnover.csv")

df.head()
```

## 2. Feature and Target Separation

The target variable is separated from the remaining columns.

```python
X = df.drop('Employee_Turnover', axis=1)
y = df['Employee_Turnover']
```

## 3. Dataset Characteristics

Inspection of the supplied dataset shows:

* 1,350 records.
* 15 input features.
* 1 binary target variable.
* No missing values in the supplied CSV file.
* Numerical values in all input columns.

The categorical features, such as employee role and department, are already represented numerically in the supplied dataset.

> The current notebook does not include a separate EDA section, additional encoding, or feature scaling. The models are trained using the supplied numerical features directly.

---

# ✂️ Train-Test Split

The dataset is divided into training and testing subsets using Scikit-learn's `train_test_split()` function.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## Training Dataset

The training dataset is used to train the Logistic Regression models.

## Testing Dataset

The testing dataset is used to evaluate how well the trained models predict employee turnover on unseen records.

### Dataset Distribution

| Dataset      | Percentage | Number of Records |
| ------------ | ---------: | ----------------: |
| Training Set |        80% |             1,080 |
| Testing Set  |        20% |               270 |
| Total        |       100% |             1,350 |

The `random_state=42` parameter ensures reproducibility of the train-test split.

---

# 🤖 Logistic Regression Models

Three Logistic Regression models are implemented to compare the impact of regularization on employee turnover prediction.

## 1. Baseline Logistic Regression

Baseline Logistic Regression serves as the reference model for evaluating the performance of the regularized models.

It estimates the probability that an employee will leave the company based on the input features.

The model uses the default regularization configuration provided by Scikit-learn.

### Implementation

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=200)

lr.fit(X_train, y_train)
```

The trained model generates predictions using:

```python
y_pred = lr.predict(X_test)
```

### Purpose

* Establish baseline classification performance.
* Provide a reference for comparing regularized models.
* Evaluate how effectively Logistic Regression predicts employee turnover.

---

## 2. Logistic Regression with L1 Regularization

L1 Regularization, also known as Lasso regularization, adds a penalty based on the absolute values of the model coefficients.

It can shrink less useful feature coefficients to zero, potentially producing a sparser model.

### Implementation

```python
from sklearn.linear_model import LogisticRegression

lasso = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    C=0.5
)

lasso.fit(X_train, y_train)
```

### Key Characteristics

* Applies an L1 penalty to the model coefficients.
* Can reduce the influence of less useful features.
* Can produce sparse models by setting some coefficients to zero.
* Uses `C=0.5` to control the regularization strength.

In Logistic Regression, a smaller value of `C` corresponds to stronger regularization.

### Purpose

To determine whether L1 Regularization improves classification performance compared with the baseline model.

---

## 3. Logistic Regression with L2 Regularization

L2 Regularization, also known as Ridge regularization, penalizes the squared magnitudes of model coefficients.

It discourages excessively large coefficients and can help control model complexity.

### Implementation

```python
from sklearn.linear_model import LogisticRegression

ridge = LogisticRegression(
    penalty='l2',
    C=1,
    max_iter=200
)

ridge.fit(X_train, y_train)
```

### Key Characteristics

* Applies an L2 penalty to model coefficients.
* Discourages excessively large coefficient values.
* Generally shrinks coefficients toward zero without forcing them exactly to zero.
* Uses `C=1` as the inverse regularization strength.

### Purpose

To evaluate the effect of L2 Regularization on classification performance and compare it with the baseline and L1 models.

---

# 📏 Model Evaluation

The models are evaluated using classification metrics provided by Scikit-learn.

The evaluation uses the same testing dataset for all three models.

## 1. Accuracy

Accuracy measures the proportion of correctly classified employees among all tested employees.

It considers both correct predictions of employees who stayed and correct predictions of employees who left.

A higher accuracy indicates that the model correctly classifies a greater proportion of the test records.

## 2. Precision

Precision measures the proportion of predicted positive cases that are actually positive.

For this project, positive cases represent employees who left the company.

Higher precision means fewer employees are incorrectly flagged as leaving relative to the total number predicted to leave.

## 3. Recall

Recall measures the proportion of actual positive cases correctly identified by the model.

For employee turnover prediction, recall indicates how many employees who actually left were identified by the model.

Higher recall can be especially important when the HR department wants to identify employees who may be at risk of leaving.

## 4. F1-Score

The F1-score is the harmonic mean of precision and recall.

It provides a combined measure of these two metrics and is useful when evaluating the trade-off between identifying potential resignations and avoiding false alarms.

## Implementation

```python
from sklearn.metrics import accuracy_score, classification_report

models = {
    'Baseline': lr,
    'Lasso': lasso,
    'Ridge': ridge
}

for name, model in models.items():
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
```

---

# 📉 Model Performance Comparison

The following results were obtained by executing the supplied notebook's modeling approach against the supplied dataset using an 80:20 train-test split and `random_state=42`.

## Accuracy Comparison

| Model                        | Regularization            |   Accuracy |
| ---------------------------- | ------------------------- | ---------: |
| Baseline Logistic Regression | Default L2 regularization |     85.93% |
| L1 Logistic Regression       | L1 / Lasso                | **87.04%** |
| L2 Logistic Regression       | L2 / Ridge                |     85.93% |

The L1 model achieved the highest test accuracy among the three configurations.

## Classification Report Comparison

The following table presents the precision, recall, and F1-score for employees who stayed (`0`) and employees who left (`1`).

| Model    | Class      | Precision | Recall | F1-Score |
| -------- | ---------- | --------: | -----: | -------: |
| Baseline | Stayed (0) |      0.85 |   0.90 |     0.87 |
| Baseline | Left (1)   |      0.87 |   0.82 |     0.84 |
| L1       | Stayed (0) |      0.86 |   0.90 |     0.88 |
| L1       | Left (1)   |      0.88 |   0.83 |     0.86 |
| L2       | Stayed (0) |      0.85 |   0.90 |     0.87 |
| L2       | Left (1)   |      0.87 |   0.82 |     0.84 |

*Values are rounded to two decimal places.*

## Interpretation

### Baseline Logistic Regression

The baseline model achieved approximately 85.93% accuracy.

It correctly classified 232 out of 270 test records.

### L1 Logistic Regression

The L1-regularized model achieved approximately 87.04% accuracy.

It correctly classified 235 out of 270 test records.

Compared with the baseline model, L1 Regularization improved test accuracy by approximately 1.11 percentage points.

The model also achieved better precision, recall, and F1-score for employees who left the company.

### L2 Logistic Regression

The L2-regularized model achieved approximately 85.93% accuracy.

Its observed results matched those of the baseline model under the current configuration.

---

# 🏆 Best Model Recommendation

Based on the current experimental results, **Logistic Regression with L1 Regularization is the recommended model**.

### Reasons for Selecting L1 Regularization

* It achieved the highest test accuracy of 87.04%.
* It correctly classified 235 out of 270 test records.
* It achieved an F1-score of 0.86 for employees who left the company.
* It improved the identification of employees who left compared with the baseline model.
* L1 Regularization can produce a sparser model by reducing some feature coefficients to zero.

### Important Considerations

Accuracy alone should not determine the final model selection in an employee turnover prediction system.

Missing an employee who is likely to resign may have different business consequences from incorrectly flagging an employee who intends to stay.

Therefore, the HR department should also consider recall, precision, F1-score, false-positive rates, and the cost of incorrect predictions.

The current results identify L1 Regularization as the best-performing configuration among the three models tested. Further validation would be useful before using the model in an actual HR decision-making process.

---

# 🛠️ Technologies Used

## Programming Language

* **Python** — Programming language used to implement the Machine Learning workflow.

## Libraries

* **Pandas** — Loading and manipulating the employee dataset.
* **Scikit-learn** — Train-test splitting, Logistic Regression, regularization, and model evaluation.
* **Jupyter Notebook** — Development and experimentation environment.

---

# 📁 Project Structure

The project contains the following files:

```text
Employee-Turnover-Prediction/
│
├── employee_turnover.csv
│
├── employee_turnover_project.ipynb
│
├── requirements.txt
│
└── README.md
```

## 📄 File Description

### `employee_turnover.csv`

Contains the employee-related features and the target variable used to train and evaluate the Machine Learning models.

### `employee_turnover_project.ipynb`

The Jupyter Notebook contains the implementation of:

* Dataset loading.
* Feature and target separation.
* Train-test splitting.
* Baseline Logistic Regression.
* L1 Regularization.
* L2 Regularization.
* Employee turnover prediction.
* Model evaluation and classification reports.

### `requirements.txt`

Contains the Python libraries required to run the project.

### `README.md`

Provides project documentation, including the problem statement, dataset description, modeling approach, evaluation results, and setup instructions.

---

# ⚙️ Installation and Setup

Follow the steps below to run the project locally.

## 1. Clone the Repository

Replace `REPOSITORY-NAME` with the actual name of your GitHub repository.

```bash
git clone https://github.com/Arpittiwari2711/Employee-Turnover-Prediction-Using-Logistic-Regression.git
```

## 2. Navigate to the Project Directory

```bash
cd REPOSITORY-NAME
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

## 4. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

## 5. Install Dependencies

If the repository contains a `requirements.txt` file, install the dependencies using:

```bash
pip install -r requirements.txt
```

## 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

## 7. Open the Project Notebook

Open the following file:

```text
employee_turnover_project.ipynb
```

Run the notebook cells sequentially to train the models and reproduce the evaluation results.

---

# 📦 Requirements

The project requires the following Python libraries:

```text
pandas
scikit-learn
jupyter
```

Create a `requirements.txt` file containing the dependencies above.

Install them using:

```bash
pip install -r requirements.txt
```

---

# 🔮 Future Improvements

The current project provides a foundation for employee turnover prediction. Several improvements could make the system more reliable and useful.

## 1. Exploratory Data Analysis

Perform a detailed analysis of employee attributes and turnover patterns.

Potential analyses include:

* Employee turnover distribution.
* Job satisfaction versus turnover.
* Monthly income versus turnover.
* Work-life balance versus turnover.
* Training hours versus turnover.
* Department-wise turnover.
* Correlation analysis between numerical features.

## 2. Feature Scaling and Preprocessing

Evaluate whether feature scaling improves Logistic Regression performance.

A preprocessing pipeline could be introduced to handle numerical and categorical features appropriately.

## 3. Hyperparameter Tuning

Experiment with different regularization strengths by adjusting the `C` parameter.

Cross-validation can help identify a configuration that generalizes more reliably to unseen employees.

## 4. Feature Selection

Analyze model coefficients to understand which employee attributes contribute most strongly to predictions.

For L1 Regularization, investigate whether any coefficients are reduced to zero.

## 5. Class Imbalance Analysis

Evaluate the target distribution and consider class weights or other techniques if class imbalance affects performance.

## 6. Additional Classification Algorithms

Compare Logistic Regression with other classification algorithms, such as:

* Decision Tree Classifier.
* Random Forest Classifier.
* Support Vector Machine.
* K-Nearest Neighbors.
* Gradient Boosting Classifier.

## 7. Cross-Validation

Use cross-validation to assess model stability across different training and validation splits.

## 8. Model Deployment

Develop a simple application using Streamlit or Flask that allows HR professionals to enter employee attributes and obtain turnover predictions.

## 9. Explainability and Responsible Use

Introduce model explainability techniques and evaluate potential bias across relevant employee groups.

Predictions should support HR investigation and employee-retention initiatives rather than automatically determine employment decisions.

---

# 🎓 Learning Outcomes

This project provides practical experience with:

* Supervised Machine Learning.
* Binary classification.
* Logistic Regression.
* L1 Regularization.
* L2 Regularization.
* Model coefficient regularization.
* Train-test splitting.
* Feature and target separation.
* Classification accuracy.
* Precision and recall.
* F1-score.
* Classification reports.
* Model performance comparison.
* Selecting a model based on experimental results.
* Applying Machine Learning to an HR problem.

---

# 💡 Conclusion

This project demonstrates how Supervised Machine Learning can be applied to employee turnover prediction.

Using employee-related information, three Logistic Regression configurations were implemented and evaluated: a baseline model, an L1-regularized model, and an L2-regularized model.

The experimental results showed that the L1-regularized model achieved the highest test accuracy of **87.04%**, compared with 85.93% for both the baseline and L2 models.

These results suggest that L1 Regularization is a promising choice for the current dataset and model configurations.

The project establishes a foundation for developing more advanced employee turnover prediction systems through additional feature analysis, cross-validation, hyperparameter tuning, and responsible model deployment.

---

# 👨‍💻 Author

## Arpit Tiwari

**B.Tech — Computer Science & Engineering**

### GitHub

[Arpittiwari2711](https://github.com/Arpittiwari2711)
