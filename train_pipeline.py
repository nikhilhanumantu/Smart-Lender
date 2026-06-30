import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Set style for premium visualizations
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#f8fafc'
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid")

# Create output directories
os.makedirs('static/images/charts', exist_ok=True)
os.makedirs('templates', exist_ok=True)

print("--- Step 1: Loading Dataset ---")
df = pd.read_csv("dataset/loan_approval_dataset.csv")
print(f"Dataset shape: {df.shape}")

# Stripping whitespaces from columns and data
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

print("--- Step 2: Imputing Missing Values ---")
# Keep track of median and mode for inference preprocessing
imputation_values = {}

# Impute categorical variables with Mode
cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
for col in cat_cols:
    mode_val = df[col].mode()[0]
    # Check if mode_val is nan (pandas mode handles it, but just in case)
    if pd.isna(mode_val):
        mode_val = df[col].dropna().mode()[0]
    imputation_values[col] = mode_val
    df[col] = df[col].fillna(mode_val)

# Impute numerical variables with Median
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
for col in num_cols:
    median_val = df[col].median()
    imputation_values[col] = median_val
    df[col] = df[col].fillna(median_val)

print("Imputed Values:")
for k, v in imputation_values.items():
    print(f"  {k}: {v}")

print("--- Step 3: Encoding Categorical Features ---")
# Create mapping for categories
mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Credit_History': {1.0: 1, 0.0: 0, 1: 1, 0: 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0},
    'Loan_Status': {'Y': 1, 'N': 0}
}

df_encoded = df.copy()

# Apply mapping
for col, mapping in mappings.items():
    if col in df_encoded.columns:
        df_encoded[col] = df_encoded[col].map(mapping)
        if col != 'Loan_Status':
            # Handle any unexpected unmapped values by filling with the encoded mode
            mode_str = str(imputation_values[col])
            encoded_mode = mapping.get(imputation_values[col], mapping.get(float(imputation_values[col]) if isinstance(imputation_values[col], (int, float)) else mode_str, 0))
            df_encoded[col] = df_encoded[col].fillna(encoded_mode).astype(int)
        else:
            df_encoded[col] = df_encoded[col].fillna(0).astype(int)

# Extract Features & Target
X = df_encoded.drop(columns=['Loan_ID', 'Loan_Status'])
y = df_encoded['Loan_Status']

print("Features columns:", X.columns.tolist())

# --- Step 4: Split & Scale ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
# Fit scaler on numerical columns only
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# --- Step 5: Model Training & Evaluation ---
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
}

accuracies = {}
for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    train_pred = clf.predict(X_train_scaled)
    test_pred = clf.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    accuracies[name] = {
        'train': train_acc,
        'test': test_acc
    }
    print(f"{name}: Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

# Save the XGBoost model as the best one
best_model_name = 'XGBoost'
best_model = models[best_model_name]

# Save model, scaler, mappings, accuracies, imputation values
model_data = {
    'model': best_model,
    'scaler': scaler,
    'mappings': mappings,
    'imputation_values': imputation_values,
    'num_cols': num_cols,
    'cat_cols': cat_cols,
    'feature_order': X.columns.tolist(),
    'accuracies': accuracies
}

with open("best_model.pkl", "wb") as f:
    pickle.dump(model_data, f)
print("Saved best model to best_model.pkl")

# --- Step 6: Generate Visual Analytics ---
# Palette colors
primary_color = '#630ed4'
secondary_color = '#9e41f5'
accent_color = '#ff6b00'
chart_colors = [primary_color, secondary_color, '#00c853', '#d50000', '#ffab00']

# 1. Loan Status Distribution Chart
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Status', hue='Loan_Status', data=df, palette={ 'Y': '#00c853', 'N': '#d50000' }, legend=False)
plt.title('Loan Approval Status Distribution', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Loan Status (Y=Approved, N=Rejected)', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.tight_layout()
plt.savefig('static/images/charts/loan_status_dist.png', dpi=150)
plt.close()

# 2. Gender Distribution Chart
plt.figure(figsize=(6, 4))
sns.countplot(x='Gender', hue='Loan_Status', data=df, palette={ 'Y': '#00c853', 'N': '#d50000' })
plt.title('Loan Status by Gender', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Gender', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.legend(title='Approved?')
plt.tight_layout()
plt.savefig('static/images/charts/gender_dist.png', dpi=150)
plt.close()

# 3. Property Area Distribution Chart
plt.figure(figsize=(6, 4))
sns.countplot(x='Property_Area', hue='Loan_Status', data=df, palette={ 'Y': '#00c853', 'N': '#d50000' })
plt.title('Loan Status by Property Area', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Property Area', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.legend(title='Approved?')
plt.tight_layout()
plt.savefig('static/images/charts/property_area_dist.png', dpi=150)
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df_encoded.drop(columns=['Loan_ID']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='Purples', fmt='.2f', linewidths=0.5, cbar=True)
plt.title('Correlation Matrix of Features', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('static/images/charts/correlation_heatmap.png', dpi=150)
plt.close()

# 5. Applicant Income Distribution
plt.figure(figsize=(7, 4.5))
sns.histplot(df['ApplicantIncome'], kde=True, color=primary_color, bins=30)
plt.title('Applicant Income Distribution', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Applicant Income ($)', fontsize=10)
plt.ylabel('Density', fontsize=10)
plt.tight_layout()
plt.savefig('static/images/charts/applicant_income_dist.png', dpi=150)
plt.close()

# 6. Accuracy Comparison Bar Chart
plt.figure(figsize=(7, 4.5))
model_names = list(accuracies.keys())
test_accs = [accuracies[m]['test'] * 100 for m in model_names]
bars = plt.bar(model_names, test_accs, color=[secondary_color, '#7c3aed', '#b388ff', primary_color], width=0.5)
plt.ylim(0, 105)
plt.title('Model Test Accuracy Comparison (%)', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Accuracy (%)', fontsize=10)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('static/images/charts/accuracy_comparison.png', dpi=150)
plt.close()

# 7. Confusion Matrix Heatmap (for XGBoost)
plt.figure(figsize=(6, 5))
xgb_test_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, xgb_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['Rejected (N)', 'Approved (Y)'],
            yticklabels=['Rejected (N)', 'Approved (Y)'])
plt.title('Confusion Matrix (XGBoost Best Model)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Predicted Label', fontsize=10)
plt.ylabel('True Label', fontsize=10)
plt.tight_layout()
plt.savefig('static/images/charts/confusion_matrix.png', dpi=150)
plt.close()

# 8. Feature Importance Graph
plt.figure(figsize=(8, 5))
importances = best_model.feature_importances_
feat_names = X.columns
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=feat_names[indices], hue=feat_names[indices], palette='Purples_r', legend=False)
plt.title('Feature Importances (XGBoost)', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Relative Importance', fontsize=10)
plt.tight_layout()
plt.savefig('static/images/charts/feature_importance.png', dpi=150)
plt.close()

print("All visual charts generated and saved in static/images/charts/")
