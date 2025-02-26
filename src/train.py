# src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import time  # For measuring training duration
from preprocess import load_and_clean_data

def train_model(data_path, n_estimators=100, max_depth=None):
    # Start an MLflow run with a dynamic run name based on hyperparameters
    with mlflow.start_run(run_name=f"rf_n_estimators_{n_estimators}_max_depth_{max_depth}") as run:
        try:
            # Log parameters directly without using YAML file
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            
            # Load and preprocess data
            data = load_and_clean_data(data_path)
            if data is None:
                return

            # Prepare data for training
            features = ['Lag_1', 'Lag_2', 'Lag_3', 'Number of employees']  # Example feature columns
            target = 'Release_Category'

            X = data[features]
            y = data[target]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Start timing the model training
            start_time = time.time()

            # Initialize and train the model with dynamic hyperparameters
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Calculate the training duration
            training_duration = time.time() - start_time

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model Accuracy: {accuracy * 100:.2f}%")

            # Log metrics (e.g., accuracy)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_duration", training_duration)  # Log duration as a metric

            # Log the model to MLflow (as a sklearn model)
            mlflow.sklearn.log_model(model, "random_forest_model")

            # Log the model locally (for saving locally)
            joblib.dump(model, 'models/my_model.pkl')

            # Optionally, log the model training duration (in seconds)
            mlflow.log_param("model_training_duration_seconds", training_duration)

        except Exception as e:
            print(f"Error during training: {e}")
            mlflow.log_param("error", str(e))


if __name__ == "__main__":
    # Experiment 1
    train_model('data/raw/mydata.xlsx', n_estimators=200, max_depth=10)

    # Experiment 2
    train_model('data/raw/mydata.xlsx', n_estimators=150, max_depth=5)

    # Experiment 3
    train_model('data/raw/mydata.xlsx', n_estimators=100, max_depth=None)
