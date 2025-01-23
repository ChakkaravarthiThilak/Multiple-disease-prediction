import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Function to load and clean data
def load_and_clean_data(file):
    df = pd.read_csv(file)
    # Remove the 'name' column if it exists
    if "name" in df.columns:
        df = df.drop(columns=["name"])
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

# Function to perform data transformation
def transform_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Function to tune the model using GridSearchCV
def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Streamlit UI and interactions
def run_app():
    st.title("Parkinson's Disease Prediction")

    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = load_and_clean_data(uploaded_file)

        # Show the data
        st.write("### Dataset Overview")
        st.dataframe(df.head())

        # User selects the target column
        target_column = st.selectbox("Select the target column", df.columns)

        # Perform data transformation (train-test split and scaling)
        X_train, X_test, y_train, y_test = transform_data(df, target_column)

        # Hyperparameter tuning for KNN
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'metric': ['euclidean', 'manhattan']
        }
        
        model = KNeighborsClassifier()  # Best model based on previous results
        best_model = tune_model(model, param_grid, X_train, y_train)
        
        # Make predictions and evaluate the model
        preds = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

        # Classification report and confusion matrix
        st.write("### Classification Report:")
        st.text(classification_report(y_test, preds))
        
        st.write("### Confusion Matrix:")
        cm = confusion_matrix(y_test, preds)
        
        # Create a figure and axis for the confusion matrix plot
        fig, ax = plt.subplots()  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1], ax=ax)
        st.pyplot(fig)  # Pass the figure to st.pyplot

if __name__ == "__main__":
    run_app()
