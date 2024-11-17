import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from river import drift

# Load the dataset (Replace with actual dataset link or file path)
df = pd.read_csv('dataset.csv')  # Replace with your actual dataset link/file path

# Display the first few rows to understand the data
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Handle missing values:
# 1. For numeric columns, fill with the mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# 2. For categorical columns, fill with the mode (most frequent value)
categorical_cols = df.select_dtypes(include=[object]).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Check if there are any missing values after imputation
print("\nMissing values after imputation:")
print(df.isnull().sum())

# **Remove non-numeric columns from correlation calculation**:
# Ensure only numeric columns are used for correlation calculation
numeric_df = df.select_dtypes(include=[np.number])

# Correlation heatmap to explore feature relationships
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Encode categorical variables if needed (assuming 'Region' is categorical)
df = pd.get_dummies(df, drop_first=True)

# Split the data into training and testing sets (assuming 'Churn' is the target variable)
train_data = df[df['Year'] < 2022]  # Assuming the 'Year' column is available
test_data = df[df['Year'] == 2022]  # Last year for testing

X_train = train_data.drop(columns=['Churn', 'Year'])
y_train = train_data['Churn']

X_test = test_data.drop(columns=['Churn', 'Year'])
y_test = test_data['Churn']

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Feature scaling (important for models like Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Display class distribution after resampling
print("\nClass distribution in training set after resampling:")
print(y_train_res.value_counts())

# Train a Logistic Regression model as an example
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train_scaled, y_train_res)

# Evaluate model on the test set
y_pred_lr = model_lr.predict(X_test_scaled)
print("\nLogistic Regression Performance:")
print(classification_report(y_test, y_pred_lr))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_lr))

# Alternatively, you can use a Decision Tree or Random Forest
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_scaled, y_train_res)

y_pred_dt = model_dt.predict(X_test_scaled)
print("\nDecision Tree Performance:")
print(classification_report(y_test, y_pred_dt))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_dt))

# Time-weighted learning: We can use a weighted loss function to emphasize more recent data

# For online learning (e.g., SGD), we can use Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

model_sgd = SGDClassifier(loss="log", random_state=42)
model_sgd.fit(X_train_scaled, y_train_res)

y_pred_sgd = model_sgd.predict(X_test_scaled)
print("\nSGD Classifier Performance:")
print(classification_report(y_test, y_pred_sgd))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_sgd))

# Alternatively, we can create an ensemble model combining different classifiers
ensemble_model = VotingClassifier(estimators=[('lr', model_lr), ('dt', model_dt), ('sgd', model_sgd)], voting='hard')
ensemble_model.fit(X_train_scaled, y_train_res)

y_pred_ensemble = ensemble_model.predict(X_test_scaled)
print("\nEnsemble Model Performance:")
print(classification_report(y_test, y_pred_ensemble))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_ensemble))

# Compare the performance of the models over different time periods (older vs newer data)
# A function to evaluate model performance over different periods:
def evaluate_performance(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred)
    print(f'AUC-ROC: {auc_roc}')
    print(classification_report(y_test, y_pred))
    return auc_roc

# Example usage on different time slices:
auc_lr_2021 = evaluate_performance(model_lr, X_train_scaled, y_train_res, X_test_scaled, y_test)
auc_lr_2022 = evaluate_performance(model_lr, X_train_scaled, y_train_res, X_test_scaled, y_test)

# Drift Detection with River (Optional but Advanced)
# Assuming you have a stream of predictions (e.g., for each customer), detect concept drift.
adwin = drift.ADWIN()
for true_value, prediction in zip(y_test, y_pred_ensemble):
    adwin.add_element(prediction)  # Add each prediction to the ADWIN detector
    if adwin.detected_change():
        print("Concept Drift Detected")

