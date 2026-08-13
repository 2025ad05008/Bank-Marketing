
# Bank Marketing Subscription Prediction

## Problem Statement

The objective of this project is to develop a Machine Learning classification system
to predict whether a bank customer will subscribe to a term deposit.

The project uses the Bank Marketing dataset from the UCI Machine Learning Repository
and compares five different Machine Learning classification algorithms.

The target variable is:

- `y = yes` → Customer subscribes to a term deposit
- `y = no` → Customer does not subscribe to a term deposit

The trained Machine Learning models are integrated into a Streamlit web application.
The application allows a user to upload a CSV file, select a Machine Learning model,
and obtain predictions for the customers in the uploaded dataset.

---

## Dataset Description

The project uses the **Bank Marketing dataset** obtained from the
**UCI Machine Learning Repository**.

The dataset contains information related to customer demographics, previous
marketing campaigns, contact information and economic indicators.

The dataset contains **20 input features** and one target variable.

### Input Features

The main input features include:

- age
- job
- marital
- education
- default
- housing
- loan
- contact
- month
- day_of_week
- duration
- campaign
- pdays
- previous
- poutcome
- emp.var.rate
- cons.price.idx
- cons.conf.idx
- euribor3m
- nr.employed

### Target Variable

The target variable is:

- `yes` – customer subscribed to a term deposit
- `no` – customer did not subscribe to a term deposit

The complete dataset used during model development is:

- `bank-additional-full.csv` – contains 41,188 examples and 20 input features.

The data was preprocessed before training the Machine Learning models.
Categorical variables were label encoded and numerical features were scaled
where required.

---

## GitHub Repository

The complete project is maintained in the following GitHub repository:

**GitHub Repository:**

https://github.com/2025ad05008/Bank-Marketing

The repository contains the Machine Learning notebook, trained models,
preprocessing files, Streamlit application, requirements file,
test data and project documentation.

---

## Machine Learning Models Used

Five Machine Learning classification models were developed and evaluated:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble)

The trained models and preprocessing objects are saved using Joblib and are
used by the Streamlit application for prediction.

---

## Data Preprocessing

The following preprocessing steps were performed before model training:

1. The dataset was separated into input features and the target variable.
2. Categorical features were converted into numerical values using
   `LabelEncoder`.
3. The target variable was encoded into numerical form for model training.
4. The dataset was divided into training and testing sets using an 80:20 split.
5. Stratified splitting was used to preserve the class distribution.
6. `StandardScaler` was used to scale the features required by models such as
   Logistic Regression, kNN and Naive Bayes.
7. The same saved encoders and scaler are used during prediction in the
   Streamlit application.

The training-test split produced:

- Training samples: **32,950**
- Testing samples: **8,238**
- Input features: **20**

---

## Model Evaluation

The models were evaluated using the following performance metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model                 | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------------ | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression      |   0.8569 | 0.9387 |    0.4347 | 0.8998 |   0.5862 | 0.5617 |
| Decision Tree            |   0.8956 | 0.7535 |    0.5343 | 0.5700 |   0.5516 | 0.4929 |
| kNN                      |   0.9053 | 0.8617 |    0.6267 | 0.3944 |   0.4841 | 0.4491 |
| Naive Bayes              |   0.8536 | 0.8606 |    0.4024 | 0.6175 |   0.4872 | 0.4189 |
| Random Forest (Ensemble) |   0.9205 | 0.9491 |    0.6898 | 0.5345 |   0.6023 | 0.5645 |

---

## Observations on Model Performance

### Logistic Regression

Logistic Regression achieved an accuracy of **0.8569** and an AUC of
**0.9387**.

It achieved the **highest recall of 0.8998** among all five models, meaning
that it was highly effective at identifying customers who actually subscribed
to the term deposit.

However, its precision was relatively low at **0.4347**, indicating that a
considerable number of positive predictions were false positives.

Overall, Logistic Regression performed particularly well when recall was
considered important.

### Decision Tree

The Decision Tree achieved an accuracy of **0.8956**.

Its AUC was **0.7535**, which was the lowest AUC among the five models.

Precision and recall were relatively balanced at **0.5343** and **0.5700**,
respectively. The model achieved reasonable accuracy but did not perform as
well as the other models in terms of AUC and MCC.

### k-Nearest Neighbors (kNN)

kNN achieved an accuracy of **0.9053** and the **highest precision of 0.6267**
among the models.

However, its recall was relatively low at **0.3944**. This indicates that
although its positive predictions were comparatively precise, it failed to
identify a significant number of customers who actually subscribed.

Its F1 score and MCC were also lower than those of Logistic Regression and
Random Forest.

### Naive Bayes

Naive Bayes achieved an accuracy of **0.8536** and an AUC of **0.8606**.

Its recall was **0.6175**, which was higher than Decision Tree and kNN, while
its precision was relatively low at **0.4024**.

Its F1 score of **0.4872** and MCC of **0.4189** were lower than those of the
stronger-performing models.

### Random Forest (Ensemble)

Random Forest achieved the **highest accuracy of 0.9205** and the
**highest AUC of 0.9491** among all five models.

It also achieved the:

- Highest precision: **0.6898**
- Highest F1 Score: **0.6023**
- Highest MCC: **0.5645**

Although its recall of **0.5345** was lower than Logistic Regression and
Naive Bayes, Random Forest provided the strongest overall balance across
the major evaluation metrics.

Therefore, **Random Forest was selected as the best-performing model for
this dataset**.

---

## Overall Winner

### Random Forest (Ensemble)

Random Forest is selected as the overall best model because it provides the
best overall combination of:

- Accuracy: **0.9205**
- AUC: **0.9491**
- Precision: **0.6898**
- F1 Score: **0.6023**
- MCC: **0.5645**

Although Logistic Regression achieved a higher recall (**0.8998**) than
Random Forest (**0.5345**), Random Forest performed better on the majority
of the evaluation metrics and therefore was selected as the overall winner.

---

## Streamlit Web Application

The trained Machine Learning models are integrated into a Streamlit web
application.

The application provides the following functionality:

1. Select a Machine Learning model.
2. Upload a CSV test dataset.
3. Display the uploaded customer data.
4. Automatically preprocess categorical and numerical features.
5. Apply the same preprocessing objects used during model training.
6. Generate predictions using the selected Machine Learning model.
7. Display predictions as **Yes** or **No**.
8. Display evaluation metrics when the uploaded dataset contains the target
   variable.
9. Display a confusion matrix.
10. Display a classification report.
11. Download the prediction results as a CSV file.

The application keeps the original uploaded dataset unchanged for display and
uses a separate processed copy internally for Machine Learning prediction.

---

## Model Files

The trained Machine Learning models and preprocessing objects are saved using
Joblib.

The following model files are generated:

- `logistic_regression.pkl`
- `decision_tree.pkl`
- `knn.pkl`
- `naive_bayes.pkl`
- `random_forest.pkl`

The preprocessing files are:

- `scaler.pkl`
- `label_encoders.pkl`
- `target_encoder.pkl`

These saved files allow the Streamlit application to load the trained models
and apply the same preprocessing used during model development.

---

## Project Structure

The main project structure is:

```text
Bank-Marketing/
│
├── app.py
├── model.py
├── README.md
├── requirements.txt
├── test_data.csv
├── model_comparison.csv
├── 2025ad05008.py
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    └── target_encoder.pkl
