import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import random
from scipy import stats
import pickle

# Set global random seed for reproducibility
np.random.seed(42)
random.seed(42)

def define_problem():
    problem_statement = "Predict the likelihood of liver disease based on patient medical records."
    assumptions = ["All features except 'Gender' are numeric.", "Dataset may be imbalanced."]
    current_solution = "Manual diagnosis based on liver function test results."
    benefits = "Early detection of liver disease, improving patient outcomes."
    return problem_statement, assumptions, current_solution, benefits

def perform_eda(df, target_column):
    print("===== Exploratory Data Analysis =====\n")
    print("Data Overview:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nClass Distribution:")
    print(df[target_column].value_counts(normalize=True) * 100)
    sns.countplot(data=df, x=target_column)
    plt.title("Class Distribution")
    plt.show()
    
    print("\n===== Correlation Heatmap =====")
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title("Correlation Matrix")
    plt.show()

def clean_data(df):
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    df['Dataset'] = df['Dataset'].replace(2, 0)  # Correct target variable

    imputer = SimpleImputer(strategy='median')  # Using 'median' instead of 'mean'
    df.iloc[:, :] = imputer.fit_transform(df)
    
    return df

def remove_outliers(df):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=['number'])))
    df = df[(z_scores < 2.5).all(axis=1)]  # Reducing threshold from 3 to 2.5
    return df

def feature_engineering(df):
    df['Albumin_Ratio'] = df['Albumin'] / (df['Total_Protiens'] + 1e-5)
    
    if 'SGPT' in df.columns and 'SGOT' in df.columns:
        df['SGPT_SGOT_Ratio'] = df['SGPT'] / (df['SGOT'] + 1e-5)
    else:
        print("Warning: 'SGPT' or 'SGOT' column not found in dataset. Skipping SGPT_SGOT_Ratio computation.")
    
    return df

def sample_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    smote = SMOTE(sampling_strategy=0.8, random_state=42)
    X, y = smote.fit_resample(X, y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def transform_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def present_results(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        results[name] = accuracy
        print(f"{name}: {accuracy:.4f}")
    
    print("\n===== Saving Best Model (XGBoost) =====")
    best_model = models['XGBoost']
    with open("Liver_disease_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print("Model saved successfully as Liver_disease_model.pkl")

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/ELCOT/indian_liver_patient.csv")
    target = "Dataset"
    
    df = clean_data(df)
    df = remove_outliers(df)
    df = feature_engineering(df)
    perform_eda(df, target_column=target)
    
    X_train, X_test, y_train, y_test = sample_data(df, target)
    X_train, X_test = transform_data(X_train, X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=700, max_depth=30, min_samples_split=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=500, learning_rate=0.03, max_depth=9, random_state=42),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost': XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=10, random_state=42)
    }
    
    present_results(models, X_train, y_train, X_test, y_test)
