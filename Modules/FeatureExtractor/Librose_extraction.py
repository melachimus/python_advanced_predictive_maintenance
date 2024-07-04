import os
import librosa
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

class FeatureExtractor:
    def __init__(self):
        self.create_output_file()

    def extract_features_from_audio(self, audio_folder):
        normal_folder = os.path.join(audio_folder, 'normal')
        abnormal_folder = os.path.join(audio_folder, 'abnormal')

        # Feature extraction parameters
        settings = MinimalFCParameters()

        # Process normal audio files
        for filename in os.listdir(normal_folder):
            if filename.endswith('.wav'):
                file_path = os.path.join(normal_folder, filename)
                self.extract_features_from_file(file_path, 'normal', settings)

        # Process abnormal audio files
        for filename in os.listdir(abnormal_folder):
            if filename.endswith('.wav'):
                file_path = os.path.join(abnormal_folder, filename)
                self.extract_features_from_file(file_path, 'abnormal', settings)

    def extract_features_from_file(self, file_path, label, settings):
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)

        # Window size and overlap (adjust as needed)
        window_size = 1000
        overlap = 0  # No overlap

        # Calculate number of windows
        num_windows = int(np.ceil(len(y) / window_size))

        # Initialize empty list for storing features
        all_features = []

        # Extract features from each window
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min(start_idx + window_size, len(y))

            # Extract features from current window
            window_data = y[start_idx:end_idx]
            window_features = self.extract_window_features(window_data, settings)

            # Add unique ID for each window
            window_features['id'] = i  # Simple incremental ID for each window
            all_features.append(window_features)

        # Convert list of dictionaries to DataFrame for tsfresh
        features_df = pd.DataFrame(all_features)

        # Extract features using tsfresh
        extracted_features = extract_features(features_df, column_id="id", default_fc_parameters=settings)

        # Add label column
        extracted_features['label'] = label

        # Save features to file
        self.append_to_csv(extracted_features)

        print(f"Extracted features for '{os.path.basename(file_path)}' ({label})")

    def extract_window_features(self, window_data, settings):
        # Extract features from a single window using tsfresh
        return {'mean': np.mean(window_data)}

    def create_output_file(self):
        # Define path to save CSV file
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_script_path, '..', '..'))
        csv_storage_path = os.path.join(base_path, 'CSV_Features')
        os.makedirs(csv_storage_path, exist_ok=True)
        self.output_file = os.path.join(csv_storage_path, 'extracted_features_new.csv')

    def append_to_csv(self, df):
        # Append DataFrame to CSV file
        if not os.path.exists(self.output_file):
            df.to_csv(self.output_file, index=False)
        else:
            df.to_csv(self.output_file, mode='a', header=False, index=False)

# Example usage:
if __name__ == "__main__":
    audio_folder = r'D:\0_dB_pump\pump\id_00'

    extractor = FeatureExtractor()
    extractor.extract_features_from_audio(audio_folder)

