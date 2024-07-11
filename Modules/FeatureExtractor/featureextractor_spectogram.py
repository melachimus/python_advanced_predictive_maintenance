import os
import pandas as pd
from tsfresh import extract_features

class FeatureExtractor:
    def __init__(self, input_file):
        self.input_file = input_file
        self.create_output_file()  
        
    def clean_data(self, data):
        if 'frequency' in data.columns:
            frequency_data = data[['frequency']]
        else:
            frequency_data = pd.DataFrame()
        
        frequency_data = frequency_data.fillna(frequency_data.mean())
        
        # Adding id column to number the rows
        frequency_data['id'] = range(len(frequency_data))
        
        return frequency_data

    def create_output_file(self):
        # Define path to save CSV file
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_script_path, '..', '..'))
        csv_storage_path = os.path.join(base_path, 'CSV_Features')
        os.makedirs(csv_storage_path, exist_ok=True)
        self.output_file = os.path.join(csv_storage_path, 'extracted_features_frequency.csv')

        # Create an empty DataFrame with column names and save it to the CSV file
        initial_data = {
            'id': [],
            'Label': [],
            'file_name': [],
            'frequency__mean': [],
            'frequency__median': [],
            'frequency__minimum': [],
            'frequency__maximum': []
        }
        initial_df = pd.DataFrame(initial_data)
        initial_df.to_csv(self.output_file, index=False)

        print(f"Empty CSV file created: {self.output_file}")

    def process_data(self):
        # Load the entire input data
        try:
            data = pd.read_csv(self.input_file)
        except Exception as e:
            print(f"Error reading input file: {str(e)}")
            return

        if 'Label' not in data.columns:
            print("No 'Label' column found in the input data.")
            return

        # Group by 'file_name' and 'Label'
        grouped_data = data.groupby(['file_name', 'Label'])

        current_id = 0  # Starting value for the id assignment

        for (file_name, label), group in grouped_data:
            cleaned_data = self.clean_data(group)

            if not cleaned_data.empty:
                custom_fc_parameters = {
                    'mean': None,
                    'median': None,
                    'minimum': None,
                    'maximum': None
                }

                try:
                    extracted_features = extract_features(
                        cleaned_data,
                        column_id='id',  
                        column_value='frequency',
                        default_fc_parameters=custom_fc_parameters
                    )

                    # Aggregate the features by taking the mean over all columns
                    aggregated_features = extracted_features.mean().to_frame().T 

                    # Add additional columns for file_name and Label
                    aggregated_features['file_name'] = file_name
                    aggregated_features['Label'] = label

                    # Add an 'id' column and number the rows
                    aggregated_features['id'] = current_id
                    current_id += 1  # Increment the id for the next group

                    # Reorder the columns in the desired order
                    columns_ordered = ['id', 'Label', 'file_name', 'frequency__mean', 'frequency__median',
                                       'frequency__minimum',
                                       'frequency__maximum']
                    aggregated_features = aggregated_features[columns_ordered]

                    # Append the aggregated features to the output file
                    aggregated_features.to_csv(self.output_file, mode='a', header=False, index=False)

                    # Print the progress
                    print(f"Processed file_name: {file_name}, Label: {label}")

                except Exception as e:
                    print(f"Error processing data for file_name: {file_name}, Label: {label}: {str(e)}")

            else:
                print(f"No 'frequency' column found in the data for file_name: {file_name}, Label: {label}.")

        print("Feature extraction completed and saved to", self.output_file)

if __name__ == '__main__':
    input_file = r'C:\Users\CR\python_advanced_predictive_maintenance\CSV\spectrogram_dfs.csv'

    extractor = FeatureExtractor(input_file)
    extractor.process_data()
