import os
import pandas as pd
from tsfresh import extract_features

# ID00 (ChatGpt)
class FeatureExtractor:
    """
    A class used to extract features from input data and save the results to a CSV file.

    Attributes
    ----------
    input_file : str
        The path to the input CSV file.
    output_file : str
        The path where the output CSV file will be saved.

    Methods
    -------
    clean_data(data):
        Cleans the data by filling missing values and adding an 'id' column.
    create_output_file():
        Creates an empty CSV file to store extracted features.
    process_data():
        Processes the input data to extract features and saves them to the output file.
    """

    def __init__(self, input_file):
        """
        Initializes the FeatureExtractor with the path to the input file.

        Parameters
        ----------
        input_file : str
            The path to the input CSV file.
        """
        self.input_file = input_file
        self.create_output_file()  
        
    def clean_data(self, data):
        """
        Cleans the input data by handling missing values and adding an 'id' column.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be cleaned.

        Returns
        -------
        pd.DataFrame
            The cleaned data with an 'id' column.
        """
        if 'amplitude' in data.columns:
            amplitude_data = data[['amplitude']]
        else:
            amplitude_data = pd.DataFrame()
        
        amplitude_data = amplitude_data.fillna(amplitude_data.mean())
        
        # Adding id column to number the rows
        amplitude_data['id'] = range(len(amplitude_data))
        
        return amplitude_data

    def create_output_file(self):
        """
        Creates an empty CSV file to store extracted features and sets the output file path.
        """
        # Define path to save CSV file
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_script_path, '..', '..'))
        csv_storage_path = os.path.join(base_path, 'CSV_Features')
        os.makedirs(csv_storage_path, exist_ok=True)
        self.output_file = os.path.join(csv_storage_path, 'extracted_features_amplitude.csv')

        # Create an empty DataFrame with column names and save it to the CSV file
        initial_data = {
            'id': [],
            'Label': [],
            'file_name': [],
            'amplitude__mean': [],
            'amplitude__median': [],
            'amplitude__minimum': [],
            'amplitude__maximum': []
        }
        initial_df = pd.DataFrame(initial_data)
        initial_df.to_csv(self.output_file, index=False)

        print(f"Empty CSV was created: {self.output_file}")

    def process_data(self):
        """
        Processes the input data to extract features and saves them to the output file.
        """
        # Load the entire input data
        try:
            data = pd.read_csv(self.input_file)
        except Exception as e:
            print(f"Error reading input file: {str(e)}")
            return

        if 'Label' not in data.columns:
            print("No 'Label' column found in input data.")
            return

        # Group by 'file_name' and 'Label'
        grouped_data = data.groupby(['file_name', 'Label'])

        current_id = 0  # Starting value for id assignment

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
                        column_value='amplitude',
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
                    columns_ordered = ['id', 'Label', 'file_name', 'amplitude__mean', 'amplitude__median',
                                       'amplitude__minimum', 'amplitude__maximum']
                    aggregated_features = aggregated_features[columns_ordered]

                    # Append the aggregated features to the output file
                    aggregated_features.to_csv(self.output_file, mode='a', header=False, index=False)

                    # Print the progress
                    print(f"Processed file_name: {file_name}, Label: {label}")

                except Exception as e:
                    print(f"Error processing data for file_name: {file_name}, Label: {label}: {str(e)}")

            else:
                print(f"No 'amplitude' column found in the data for file_name: {file_name}, Label: {label}.")

        print("Feature extraction completed and saved to", self.output_file)








