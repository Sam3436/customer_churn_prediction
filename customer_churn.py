import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier

# data loading

try:
    df = pd.read_csv('customer-churn.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Could not find 'customer-churn.csv'. Using a simplified dummy structure for demonstration.")
    data = {
        'customerID': range(1, 1001),
        'Gender': np.random.choice(['Male', 'Female'], 1000),
        'SeniorCitizen': np.random.randint(0, 2, 1000),
        'Partner': np.random.choice(['Yes', 'No'], 1000),
        'Dependents': np.random.choice(['Yes', 'No'], 1000),
        'Tenure': np.random.randint(1, 73, 1000),
        'MonthlyCharges': np.random.uniform(20, 120, 1000),
        'TotalCharges': np.random.uniform(25, 8000, 1000),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
        'Churn': np.random.choice(['Yes', 'No'], 1000, p=[0.26, 0.74]) # Simulating imbalance
    }
    df = pd.DataFrame(data)

# Convert the target column to 0s and 1s
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Initial cleanup for common issues (Telco data specific)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']) # Drop rows where TotalCharges were missing

print(f"\nTotal rows after cleaning: {len(df)}")
print(f"Churn Rate: {df['Churn'].mean() * 100:.2f}%")

# --- 3. Define Feature Groups and Transformation ---

# Drop the Customer ID column
df = df.drop('customerID', axis=1, errors='ignore')

# Separate features (X) and target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Define feature types for preprocessing
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# --- Preprocessing Steps ---
# 1. Numerical Pipeline: Scale the numerical features
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# 2. Categorical Pipeline: One-Hot Encode the categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 3. Create the Column Transformer (Combines the two pipelines)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep any other columns if defined (not necessary here)
)

# --- Train-Test Split ---
# Use stratify=y to ensure the train and test sets have a similar churn rate
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nData split into training and testing sets.")

# --- 4. Model Training Pipeline (with SMOTE) ---

# Define the base model (XGBoost is typically excellent for this task)
# The 'scale_pos_weight' is an alternative way to handle imbalance specific to XGBoost
# For simplicity and to show SMOTE, we rely on SMOTE here.
xgb_model = XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=100,
    learning_rate=0.05
)

# Create an imbalanced-learn Pipeline:
# 1. Apply preprocessing (scaling/encoding)
# 2. Apply SMOTE to the processed training data
# 3. Fit the XGBoost model
model_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb_model)
])

print("Training the XGBoost model with SMOTE...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 5. Prediction and Probability ---
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

# --- 6. Model Evaluation ---
print("\n" + "="*50)
print("       MODEL EVALUATION (XGBoost with SMOTE)       ")
print("="*50)

# 1. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 2. AUC-ROC Score
auc_roc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC Score: {auc_roc:.4f}")

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Retained (0)', 'Churned (1)'], 
            yticklabels=['Retained (0)', 'Churned (1)'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 4. ROC Curve Visualization
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.show()


# --- 7. Feature Importance ---
# We need to extract the feature names after one-hot encoding
feature_names = numerical_features + list(model_pipeline['preprocessor'].named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features))

# Extract feature importances from the trained classifier
importance = model_pipeline['classifier'].feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Visualize the top 15 features
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Feature Importances for Churn Prediction')
plt.show()


print("\n--- Top 5 Features Driving Churn ---")
print(feature_importance_df.head(5).to_markdown(index=False))

