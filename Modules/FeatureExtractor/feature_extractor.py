import pandas as pd
import os
from tsfresh import extract_features

class FeatureExtractor:
    def __init__(self, input_file, chunk_size=1000):
        self.input_file = input_file
        self.chunk_size = chunk_size
        self.output_file = None
        
    def clean_data(self, chunk):
        if 'amplitude' in chunk.columns:
            amplitude_chunk = chunk[['amplitude']]
        else:
            amplitude_chunk = pd.DataFrame()
        
        amplitude_chunk = amplitude_chunk.fillna(amplitude_chunk.mean())
        
        return amplitude_chunk

    def create_output_file(self):
        # Define path to save CSV file
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_script_path, '..', '..'))
        csv_storage_path = os.path.join(base_path, 'CSV_Features')
        os.makedirs(csv_storage_path, exist_ok=True)
        self.output_file = os.path.join(csv_storage_path, 'extracted_features.csv')

        # Create an empty DataFrame with column labels and save it to CSV file
        initial_data = {
            'Schritt': [],
            'amplitude__mean': [],
            'amplitude__median': [],
            'amplitude__variance': [],
            'amplitude__standard_deviation': [],
            'amplitude__minimum': [],
            'amplitude__maximum': [],
            'Label': []
        }
        initial_df = pd.DataFrame(initial_data)
        initial_df.to_csv(self.output_file, index=False)

        print(f"Empty CSV file created: {self.output_file}")

    def process_chunks(self):
        # Check if output file has been created already
        if not self.output_file:
            self.create_output_file()

        # Process chunks of data and extract features
        chunk_iter = pd.read_csv(self.input_file, chunksize=self.chunk_size)
        chunk_id_start = 1000

        custom_fc_parameters = {
            'mean': None,
            'median': None,
            'variance': None,
            'standard_deviation': None,
            'minimum': None,
            'maximum': None
        }

        for i, chunk in enumerate(chunk_iter):
            chunk['id'] = chunk_id_start + i * self.chunk_size
            cleaned_chunk = self.clean_data(chunk)
            cleaned_chunk['id'] = chunk['id']

            if 'Label' not in chunk.columns:
                print(f"No 'Label' column found in chunk {i}, skipping.")
                continue

            if not cleaned_chunk.empty:
                try:
                    extracted_features = extract_features(
                        cleaned_chunk,
                        column_id='id',
                        column_value='amplitude',
                        default_fc_parameters=custom_fc_parameters
                    )

                    # Insert 'Schritt' and 'Label' columns
                    extracted_features.insert(0, 'Schritt', chunk['id'].iloc[0])
                    extracted_features['Label'] = chunk['Label'].iloc[0]

                    # Save extracted features and label to CSV file
                    extracted_features.to_csv(self.output_file, index=False, mode='a', header=False)

                    print(f"Chunk {i}: Features extracted and saved to CSV file.")

                except Exception as e:
                    print(f"Error processing chunk {i}: {str(e)}")
            else:
                print(f"No 'amplitude' column found in chunk {i}, skipping.")

        print("Feature extraction completed and saved to", self.output_file)

if __name__ == '__main__':
    input_file = r'D:\ML\amplituden_dfs.csv'
    chunk_size = 1000

    extractor = FeatureExtractor(input_file, chunk_size)
    extractor.process_chunks()

