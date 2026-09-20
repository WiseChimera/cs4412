# ==============================================================================
# ASSIGNMENT 1: STUDENT STARTER NOTEBOOK
# Course: Data Mining (Undergrad High-Level Summer Online Course)
# Topics: Linear & Logistic Regression, Class Imbalance Handling
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. Environment Setup & Library Imports
# ------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Processing & Models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

# Evaluation Metrics
from sklearn.metrics import (
    mean_squared_error, 
    r2_score, 
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve, 
    auc
)

# Imbalance Treatment
# NOTE: If not installed, run 'pip install imbalanced-learn' in your terminal
from imblearn.over_sampling import SMOTE

# Plotting Settings
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 7)

print("Packages loaded. Ready to begin.")

# ------------------------------------------------------------------------------
# 1. Dataset Generation (Synthetic Mocking Environment)
# ------------------------------------------------------------------------------
# Set seed for reproducibility. Do not change this.
np.random.seed(42)

# --- 1A. Generate Bike Sharing Regression Dataset ---
n_samples_bike = 500
bike_data = {
    'temp': np.random.uniform(0.1, 0.9, n_samples_bike),
    'hum': np.random.uniform(0.3, 0.8, n_samples_bike),
    'windspeed': np.random.uniform(0.05, 0.5, n_samples_bike),
    'season': np.random.choice(['spring', 'summer', 'fall', 'winter'], n_samples_bike),
    'holiday': np.random.choice([0, 1], n_samples_bike, p=[0.95, 0.05])
}
df_bike = pd.DataFrame(bike_data)
df_bike['cnt'] = (1500 + (3500 * df_bike['temp']) - (800 * df_bike['hum']) - 
                  (500 * df_bike['windspeed']) + np.random.normal(0, 150, n_samples_bike)).astype(int)

# --- 1B. Generate Heart Disease Classification Dataset ---
n_samples_heart = 600
heart_data = {
    'age': np.random.randint(29, 78, n_samples_heart),
    'chol': np.random.randint(126, 564, n_samples_heart),
    'thalach': np.random.randint(71, 202, n_samples_heart),
    'sex': np.random.choice([0, 1], n_samples_heart, p=[0.3, 0.7]),
    'cp': np.random.choice([0, 1, 2, 3], n_samples_heart)
}
df_heart = pd.DataFrame(heart_data)
log_odds = -5 + (0.04 * df_heart['age']) - (0.03 * df_heart['thalach']) + (1.2 * df_heart['sex'])
prob = 1 / (1 + np.exp(-log_odds))
df_heart['target'] = np.where(prob > np.percentile(prob, 85), 1, 0)

print(f"Loaded df_bike Shape: {df_bike.shape}")
print(f"Loaded df_heart Shape: {df_heart.shape}")


# ==============================================================================
# TASK 1: EXPLORATORY DATA ANALYSIS & PREPROCESSING
# ==============================================================================
print("\n--- Running Task 1 ---")

# 1.1 One-Hot Encode Categorical Columns using pd.get_dummies()
# Hint: Encode 'season' for df_bike and 'cp' for df_heart. Use drop_first=True.
# TODO: Your code here
df_bike_encoded = None
df_heart_encoded = None

# 1.2 Separate Features (X) and Target (y) variables
# TODO: Your code here
X_bike = None
y_bike = None

X_heart = None
y_heart = None

# 1.3 Split both datasets into 80% training and 20% testing sets
# Hint: Use train_test_split, set random_state=42. Stratify the heart dataset target.
# TODO: Your code here
X_train_b, X_test_b, y_train_b, y_test_b = None, None, None, None
X_train_h, X_test_h, y_train_h, y_test_h = None, None, None, None

# 1.4 Standardize continuous features using StandardScaler
# Hint: Fit and transform on train data; transform ONLY on test data.
# TODO: Your code here
X_train_b_scaled = None
X_test_b_scaled = None

X_train_h_scaled = None
X_test_h_scaled = None


# ==============================================================================
# TASK 2: MULTIPLE LINEAR REGRESSION MODEL
# ==============================================================================
print("\n--- Running Task 2 ---")

# 2.1 Initialize and fit the Multiple Linear Regression model
# TODO: Your code here
lin_reg = None

# 2.2 Print out the model intercept and feature coefficients
# TODO: Your code here

# 2.3 Generate predictions on the scaled test set
# TODO: Your code here
y_pred_b = None

# 2.4 Compute and print regression metrics: MSE, RMSE, and R2 Score
# TODO: Your code here


# ==============================================================================
# TASK 3: LOGISTIC REGRESSION MODEL & CLASS IMBALANCE MITIGATION
# ==============================================================================
print("\n--- Running Task 3 ---")

# 3.1 Print the percentage distribution of the target variable in y_train_h
# TODO: Your code here

# 3.2 Initialize SMOTE and resample the scaled training classification data
# Hint: Use the fit_resample method on X_train_h_scaled and y_train_h with random_state=42.
# TODO: Your code here
X_train_h_smote, y_train_h_smote = None, None

# 3.3 Set up three variants of the Logistic Regression model
# Variant 1: Baseline model (default settings)
# Variant 2: Algorithm-level balance (set class_weight='balanced')
# Variant 3: Data-level balance (trained on SMOTE resampled data)
# Hint: Remember to use random_state=42 for all initializations.
# TODO: Your code here
models_logistic = {
    'Baseline': None,
    'Balanced_Weights': None,
    'SMOTE_Resampled': None
}

# 3.4 Train the models, generate predictions, and extract performance metrics
# Create storage structures for mapping evaluation metrics
metrics_summary = {}
roc_plotting_data = {}

for name, model in models_logistic.items():
    # TODO: Fit the model on the correct training data variation
    pass
    
    # TODO: Generate class predictions on the original unmutated test set (X_test_h_scaled)
    preds = None
    
    # TODO: Generate predicted probabilities using predict_proba() for the positive class (column 1)
    probs = None
    
    # TODO: Compute and store classification metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
    # Complete the dictionary entry below
    metrics_summary[name] = {
        'Accuracy': None,
        'Precision': None,
        'Recall': None,
        'F1_Score': None,
        'Conf_Matrix': None
    }
    
    # TODO: Compute roc_curve and auc values, then save them into roc_plotting_data
    fpr, tpr, thresholds = None, None, None
    roc_auc = None
    roc_plotting_data[name] = (fpr, tpr, roc_auc)

# 3.5 Print out your metrics summary block for all three configurations
# TODO: Your code here


# ==============================================================================
# TASK 4: VISUALIZATION
# ==============================================================================
print("\n--- Running Task 4 ---")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 4.1 Subplot 1: Create a scatter plot of Actual vs. Predicted values for Linear Regression
# Hint: Draw a diagonal line indicating perfect prediction accuracy.
# TODO: Your code here

# 4.2 Subplot 2: Plot and overlay ROC curves for Baseline, Balanced, and SMOTE models
# Hint: Iterate through roc_plotting_data to label curves and include the calculated AUC score.
# TODO: Your code here

plt.tight_layout()
plt.show()
