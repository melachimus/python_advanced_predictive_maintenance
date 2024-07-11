import os
import librosa
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

class FeatureExtractor:
    def __init__(self):
        self.create_output_file()
        self.current_id = 0  # Initialize the ID counter

    def extract_features_from_audio(self, audio_folder):
        normal_folder = os.path.join(audio_folder, 'normal')
        abnormal_folder = os.path.join(audio_folder, 'abnormal')

        # Feature extraction parameters
        settings = MinimalFCParameters()

        # Process normal audio files
        self.process_folder(normal_folder, 'normal', settings)

        # Process abnormal audio files
        self.process_folder(abnormal_folder, 'abnormal', settings)

    def process_folder(self, folder_path, label, settings):
        for filename in os.listdir(folder_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                try:
                    self.extract_features_from_file(file_path, label, settings)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

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
            window_features = self.extract_window_features(window_data)

            # Add unique ID and file name for each window
            window_features['id'] = self.current_id
            window_features['file_name'] = os.path.basename(file_path)
            all_features.append(window_features)
            self.current_id += 1  # Increment the ID for each window

        # Convert list of dictionaries to DataFrame for tsfresh
        features_df = pd.DataFrame(all_features)

        # Store the IDs and file names separately
        ids = features_df['id']
        file_names = features_df['file_name']

        # Drop the ID and file name columns for tsfresh processing
        features_df = features_df.drop(columns=['id', 'file_name'])

        # Add 'id' as a column for tsfresh
        features_df['id'] = ids

        # Extract features using tsfresh
        extracted_features = extract_features(features_df, column_id='id', default_fc_parameters=settings)

        # Add label, id, and file_name columns back
        extracted_features['label'] = label
        extracted_features['id'] = ids.values
        extracted_features['file_name'] = file_names.values

        # Reorder columns to include id, file_name, and label at the beginning
        extracted_features = extracted_features[['id', 'file_name', 'label'] + [col for col in extracted_features.columns if col not in ['id', 'file_name', 'label']]]

        # Save features to file
        self.append_to_csv(extracted_features)

        print(f"Extracted features for '{os.path.basename(file_path)}' ({label})")

    def extract_window_features(self, window_data):
        # Calculate the magnitude of the window data
        magnitude = np.abs(window_data)

        # Calculate amplitude statistics
        amplitude_mean = np.mean(window_data)
        amplitude_median = np.median(window_data)
        amplitude_min = np.min(window_data)
        amplitude_max = np.max(window_data)

        # Calculate magnitude statistics
        features = {
            'mean_magnitude': np.mean(magnitude),
            'median_magnitude': np.median(magnitude),
            'minimum_magnitude': np.min(magnitude),
            'maximum_magnitude': np.max(magnitude),
            'mean_amplitude': amplitude_mean,
            'median_amplitude': amplitude_median,
            'minimum_amplitude': amplitude_min,
            'maximum_amplitude': amplitude_max
        }

        return features

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



