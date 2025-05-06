import pandas as pd
import joblib
import logging
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_undersampling import run_undersampling

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
CONFIG = {
    "data_path": "bank-additional-full.csv",
    "test_size": 0.2,
    "random_state": 42,
    "model_path": "bank_marketing_xgb_model.pkl",
}

def load_data(csv_name):
    """Load dataset from CSV."""
    try:
        df = pd.read_csv(csv_name, sep=";", encoding="utf-8")
        logging.info("Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def preprocess_data(df):
    """Preprocess dataset: encoding, feature engineering, scaling."""
    try:
        df = df.drop(columns=["duration"])  # Remove post-call feature
        
        categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
        categorical_features.remove("y")  # Exclude target
        
        df["y"] = df["y"].map({"yes": 1, "no": 0})
        df = pd.get_dummies(df, columns=categorical_features)

         # Feature engineering
        feature_interactions = {
            "age_campaign_interaction": df["age"] * df["campaign"],
            "nr.employed_euribor3m_interaction": df["nr.employed"] * df["euribor3m"],
            "age_euribor3m_interaction": df["age"] * df["euribor3m"],
            "age_nr.employed_interaction": df["age"] * df["nr.employed"],
            "euribor3m_campaign_interaction": df["euribor3m"] * df["campaign"],
            "contact_telephone_default_unknown_interaction": df["contact_telephone"] * df["default_unknown"],
            "month_oct_default_unknown_interaction": df["month_oct"] * df["default_unknown"],
            "contact_cellular_housing_yes_interaction": df["contact_cellular"] * df["housing_yes"],
            "housing_yes_default_no_interaction": df["housing_yes"] * df["default_no"]
        }

        df = df.assign(**feature_interactions)


        logging.info("Data preprocessing completed.")
        return df
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def split_data(df):
    """Split dataset into training and testing sets."""
    X = df.drop(columns=["y"])
    y = df["y"]
    return train_test_split(X, y, test_size=CONFIG["test_size"], stratify=y, random_state=CONFIG["random_state"])

def handle_class_imbalance(X_train, y_train):
    """Handle class imbalance with SMOTETomek."""
    smote_tomek = SMOTETomek(random_state=CONFIG["random_state"])
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
    logging.info("Class imbalance handled using SMOTETomek.")
    return X_train_resampled, y_train_resampled

def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """Train the XGBoost model."""
    model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.05, scale_pos_weight=3, random_state=CONFIG["random_state"])
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print results."""
    y_pred = model.predict(X_test)
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

def save_model(model, scaler):
    """Save the trained model and scaler."""
    joblib.dump((model, scaler), CONFIG["model_path"])
    logging.info(f"Model saved as {CONFIG['model_path']}.")

from typing import List, Tuple
import pandas as pd

def get_feature_importance(model, X: pd.DataFrame, top_n: int = 10) -> List[Tuple[str, float]]:
    """

    Parameters:
    - model: Trained model with `feature_importances_` attribute.
    - X: DataFrame containing feature names.
    - top_n: Number of top features to return (default is 10).

    Returns:
    - List of tuples containing feature names and their importance scores.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature importances attribute.")

    importances = model.feature_importances_
    feature_names = X.columns
    sorted_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    top_features = sorted_importances[:top_n]

    # Log the important features
    logging.info("\nTop Important Features:")
    for feature, importance in top_features:
        logging.info(f"{feature}: {importance:.4f}")

    return top_features




def main():
    """Main function to run the entire pipeline."""
    csv_name = run_undersampling(CONFIG["data_path"])
    df = load_data(csv_name)
    df = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_resampled, X_test)

    model = train_model(X_train_scaled, y_train_resampled)
    get_feature_importance(model, X_train)
    evaluate_model(model, X_test_scaled, y_test)
    save_model(model, scaler)

if __name__ == "__main__":
    main()