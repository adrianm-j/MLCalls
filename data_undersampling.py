import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_undersampling(file_path, ratio=3, output_file="bank_marketing_balanced.csv"):
    """
    Perform undersampling to balance the dataset.

    Parameters:
    - file_path (str): Path to the original dataset CSV.
    - ratio (int): Desired ratio of 'no' samples to 'yes' samples.
    - output_file (str): Name of the new undersampled dataset file.

    Returns:
    - str: Path to the new dataset file.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path, sep=";", encoding="utf-8")
        logging.info("Dataset loaded successfully.")

        # Ensure 'y' column exists
        if "y" not in df.columns:
            logging.error("Error: Column 'y' not found in the dataset.")
            return None

        # Separate classes
        yes_df = df[df['y'] == 'yes']
        no_df = df[df['y'] == 'no']

        # Check sample size limits
        yes_count = len(yes_df)
        no_sample_size = min(len(no_df), yes_count * ratio)

        if no_sample_size < yes_count:
            logging.warning("Warning: Insufficient 'no' samples to maintain the desired ratio.")

        # Resample 'no' class
        no_sampled = no_df.sample(n=no_sample_size, random_state=42)

        # Combine and shuffle
        new_df = pd.concat([yes_df, no_sampled]).sample(frac=1, random_state=42)
        new_df.to_csv(output_file, index=False, sep=";")

        logging.info(f"New dataset created and saved as '{output_file}' with a {ratio}:1 ratio.")

        return output_file
    
    except FileNotFoundError:
        logging.error(f"Error: File '{file_path}' not found.")
        return None
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    new_csv = run_undersampling("bank-additional-full.csv", ratio=3)
    if new_csv:
        logging.info(f"Undersampling completed. New dataset saved at: {new_csv}")