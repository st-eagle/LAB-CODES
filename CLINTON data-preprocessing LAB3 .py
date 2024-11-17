# Importing Libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Step 1: Data Collection
# Load the Titanic dataset from Seaborn
df = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(df.head())

# Step 2: Data Cleaning
# 2.1: Inspect for Missing Values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# 2.2: Handle Missing Values
# Drop columns with too many missing values
df = df.drop(columns=['deck', 'embark_town'])

# Impute missing values for 'age' with the median
df['age'] = df['age'].fillna(df['age'].median())

# Impute missing values for 'embarked' with the mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Check if there are any missing values left
missing_values = df.isnull().sum()
print("\nMissing Values After Imputation:\n", missing_values)

# Step 3: Handling Outliers
# 3.1: Identify Outliers using Boxplots
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['age'])
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['fare'])
plt.title('Boxplot of Fare')

plt.tight_layout()
plt.show()

# 3.2: Handle Outliers
# Cap outliers in the 'age' column at the 95th percentile
age_95th_percentile = df['age'].quantile(0.95)
df['age'] = df['age'].apply(lambda x: min(x, age_95th_percentile))

# Remove extreme outliers in the 'fare' column (above the 99th percentile)
fare_99th_percentile = df['fare'].quantile(0.99)
df = df[df['fare'] <= fare_99th_percentile]

# Check the updated distributions
sns.boxplot(x=df['age'])
plt.title('Updated Boxplot of Age')
plt.show()

sns.boxplot(x=df['fare'])
plt.title('Updated Boxplot of Fare')
plt.show()

# Step 4: Data Normalization
# Apply Min-Max Scaling to 'age' and 'fare'
scaler = MinMaxScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# Display the first few rows to see the changes
print("\nData After Normalization:\n", df[['age', 'fare']].head())

# Step 5: Feature Engineering
# 5.1: Create a family_size column by combining 'sibsp' and 'parch'
df['family_size'] = df['sibsp'] + df['parch']

# Display the first few rows to verify
print("\nFamily Size Column:\n", df[['sibsp', 'parch', 'family_size']].head())

# 5.2: Create a 'title' column by extracting titles from 'name'
df['title'] = df['name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

# Display the first few rows to verify
print("\nTitle Column:\n", df[['name', 'title']].head())

# Step 6: Feature Selection
# 6.1: Correlation Analysis
correlation_matrix = df.corr()

# Plot the heatmap of the correlation matrix
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# 6.2: Feature Importance using RandomForest
# Prepare the data for model building
df = df.drop(columns=['name', 'ticket', 'embarked', 'class', 'who', 'embark_town', 'alive', 'alone'])  # Drop non-numeric and non-essential features
X = df.drop(columns=['survived'])
y = df['survived']

# Initialize Random Forest and fit the model
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to view feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display the feature importance
print("\nFeature Importance:\n", feature_importance)

# Step 7: Model Building
# 7.1: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Evaluate the model using Classification Report, Confusion Matrix, and other metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7.2: Additional Evaluation Metrics
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred)
print("\nPrecision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("\nRecall:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("\nF1 Score:", f1)
