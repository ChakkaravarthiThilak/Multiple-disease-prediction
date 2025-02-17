import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import random
import joblib

# Set global random seed for reproducibility
np.random.seed(42)
random.seed(42)

# 1. Define the problem
def define_problem():
    problem_statement = "Predict the likelihood of Kidney Disease based on patient data."
    assumptions = ["All features except 'name' contribute to predictions.", "Dataset is balanced and well-labeled."]
    current_solution = "Manual diagnosis by specialists."
    benefits = "Early and accurate predictions to support healthcare decisions."
    return problem_statement, assumptions, current_solution, benefits

# 2. Prepare the Data - EDA
def perform_eda(df, target_column=None):
    print("===== Exploratory Data Analysis =====\n")
    
    # Data Overview
    print("Data Overview:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:\n", df.isnull().sum())
    
    # Check for class imbalance
    if target_column:
        print("\nClass Distribution:")
        print(df[target_column].value_counts(normalize=True) * 100)
        sns.countplot(data=df, x=target_column)
        plt.title("Class Distribution")
        plt.show()
    
    # Univariate Analysis
    print("\n===== Univariate Analysis =====")
    for column in df.select_dtypes(include=['float64']).columns:
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        plt.show()
    
    # Bivariate Analysis
    print("\n===== Bivariate Analysis =====")
    if target_column:
        for column in df.select_dtypes(include=['float64']).columns:
            if column != target_column:
                sns.boxplot(data=df, x=target_column, y=column)
                plt.title(f"{column} vs {target_column}")
                plt.show()
    
    # Correlation Heatmap
    print("\n===== Correlation Heatmap =====")
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Correlation Matrix")
    plt.show()

# 2.2 Data Cleaning and Transformation
def clean_data(df):
    # Handle categorical features and missing values
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column]).codes
    
    # Handle missing values using mean imputation for numeric columns
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df

def sample_data(df, target_column):
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def transform_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# 3. Spot Check Algorithms
def evaluate_algorithms(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, penalty='l2', random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, max_depth=10, n_estimators=50, min_samples_split=10),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(random_state=42, C=1.0, kernel='rbf'),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, n_estimators=50, max_depth=5, min_child_weight=10, subsample=0.8, colsample_bytree=0.8)
    }
    results = {}
    for name, model in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
        results[name] = np.mean(cv_scores)
        print(f"{name}: {np.mean(cv_scores):.4f}")
    return results

# 4. Improve Results
def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(
        model, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy'
    )
    grid_search.fit(X_train, y_train)
    print("Best Params:", grid_search.best_params_)
    return grid_search.best_estimator_

# 5. Finalize Project
def present_results(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, preds))
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.show()

# Main Workflow
if __name__ == "__main__":
    # Update to point to Kidney Disease dataset
    data_file = "C:/Users/ELCOT/kidney_disease.csv"  # Update with correct path
    df = pd.read_csv(data_file)
    
    # Update the target column name for Kidney Disease dataset
    target = "classification"  # 'classification' is the correct target column name
    
    # Clean the data
    df = clean_data(df)
    
    # Perform EDA
    perform_eda(df, target_column=target)
    
    # Split the data
    X_train, X_test, y_train, y_test = sample_data(df, target)
    
    # Data transformation
    X_train, X_test = transform_data(X_train, X_test)
    
    # Evaluate Algorithms
    results = evaluate_algorithms(X_train, y_train)
    
    # Model Training with XGBoost
    model = XGBClassifier(random_state=42)
    
    # Hyperparameter tuning
    best_model = tune_model(model, {'n_estimators': [50, 100, 150]}, X_train, y_train)
    
    # Save the trained model
    joblib.dump(best_model, "kidney_disease_model.pkl")
    print("Model saved as kidney_disease_model.pkl")
    
    # Present results
    present_results(best_model, X_test, y_test)
