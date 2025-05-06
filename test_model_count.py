import pandas as pd
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
MODEL_PATH = "bank_marketing_xgb_model.pkl"
DATA_PATH = "bank-additional-full.csv"

def load_model(model_path):
    """Load the trained model and scaler."""
    try:
        model, scaler = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        logging.error(f"Error: Model file '{model_path}' not found.")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error loading model: {e}")
        return None, None

def load_and_preprocess_data(csv_name):
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv(csv_name, sep=";", encoding="utf-8")
        logging.info("Dataset loaded successfully.")

        # Drop irrelevant features
        df = df.drop(columns=["duration"])  # Remove post-call info
        
        # Handle categorical features
        categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
        if "y" in categorical_features:
            categorical_features.remove("y")
        else:
            logging.error("Error: Target column 'y' missing.")
            return None, None
        
        # Encode target variable
        df["y"] = df["y"].map({"yes": 1, "no": 0})

        # One-hot encode categorical features
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

        logging.info("Preprocessing and feature engineering completed.")
        return df.drop(columns=["y"]), df["y"]
    
    except FileNotFoundError:
        logging.error(f"Error: Data file '{csv_name}' not found.")
        return None, None
    except Exception as e:
        logging.error(f"Unexpected error in preprocessing: {e}")
        return None, None

def scale_features(X, scaler):
    """Scale dataset using pre-loaded scaler."""
    try:
        return scaler.transform(X)
    except Exception as e:
        logging.error(f"Error scaling data: {e}")
        return None

def get_predictions(model, X_scaled):
    """Generate predictions using the trained model."""
    try:
        return model.predict(X_scaled)
    except Exception as e:
        logging.error(f"Error predicting outcomes: {e}")
        return None

def evaluate_predictions(predictions, y):
    """Evaluate prediction performance."""
    if predictions is None or y is None:
        logging.error("Error: Missing prediction data.")
        return
    
    matching_yes = ((predictions == 1) & (y == 1)).sum()
    matching_no = ((predictions == 0) & (y == 0)).sum()
    not_matching_yes = ((predictions == 0) & (y == 1)).sum()
    not_matching_no = ((predictions == 1) & (y == 0)).sum()

    matching_yes_pct = (matching_yes / max(1, (matching_yes + not_matching_yes))) * 100
    matching_no_pct = (matching_no / max(1, (matching_no + not_matching_no))) * 100
    not_matching_yes_pct = (not_matching_yes / (matching_yes + not_matching_yes)) * 100
    not_matching_no_pct = (not_matching_no / (matching_no + not_matching_no)) * 100

    logging.info(f"Matching Outcomes (Yes): {matching_yes} ({matching_yes_pct:.2f}%) of 'yes' cases")
    logging.info(f"Matching Outcomes (No): {matching_no} ({matching_no_pct:.2f}%) of 'no' cases")
    logging.info(f"Not Matching Outcomes (Yes): {not_matching_yes} ({not_matching_yes_pct:.2f}%) of 'yes' cases")
    logging.info(f"Not Matching Outcomes (No): {not_matching_no} ({not_matching_no_pct:.2f})% of 'no' cases")

def main():
    """Main pipeline function."""
    model, scaler = load_model(MODEL_PATH)
    if model is None or scaler is None:
        return
    
    X, y = load_and_preprocess_data(DATA_PATH)
    if X is None or y is None:
        return
    
    X_scaled = scale_features(X, scaler)
    if X_scaled is None:
        return

    predictions = get_predictions(model, X_scaled)
    evaluate_predictions(predictions, y)

if __name__ == "__main__":
    main()