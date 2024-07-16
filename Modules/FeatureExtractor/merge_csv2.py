
# ID:01 (ChatGPT)
import os
import pandas as pd

class CSVMerger:
    """
    A class used to merge two CSV files based on common columns and save the results to a new CSV file.

    Attributes
    ----------
    file1 : str
        The path to the first input CSV file.
    file2 : str
        The path to the second input CSV file.
    output_file : str
        The path where the merged CSV file will be saved.

    Methods
    -------
    create_output_file():
        Defines the path to save the merged CSV file.
    merge_csv_files():
        Merges the two CSV files on common columns and returns the merged DataFrame.
    save_merged_csv():
        Saves the merged DataFrame to the output CSV file.
    """

    def __init__(self, file1, file2):
        """
        Initializes the CSVMerger with the paths to the two input files.

        Parameters
        ----------
        file1 : str
            The path to the first input CSV file.
        file2 : str
            The path to the second input CSV file.
        """
        self.file1 = file1
        self.file2 = file2
        self.create_output_file()  
    
    def create_output_file(self):
        """
        Defines the path to save the merged CSV file and creates the necessary directories.
        """
        # Define the path to save the CSV file
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_script_path, '..', '..'))
        csv_storage_path = os.path.join(base_path, 'CSV_Features')
        os.makedirs(csv_storage_path, exist_ok=True)
        self.output_file = os.path.join(csv_storage_path, 'Merge_CSV.csv')

    def merge_csv_files(self):
        """
        Merges the two input CSV files on 'id' and 'Label' columns.

        Returns
        -------
        pd.DataFrame
            The merged DataFrame.
        """
        # Load the entire input data
        try:
            df1 = pd.read_csv(self.file1)
            df2 = pd.read_csv(self.file2)
        except FileNotFoundError as e:
            print(f"Error reading CSV files: {e}")
            return None
        
        # Sort the DataFrames by 'id' and 'Label'
        df1.sort_values(by=['id', 'Label'], inplace=True)
        df2.sort_values(by=['id', 'Label'], inplace=True)

        # Merge the DataFrames on 'id' and 'Label'
        merged_df = pd.merge(df1, df2, on=['id', 'Label'])

        # Sort the columns in the desired order
        columns_ordered = ['id', 'Label', 
                           'amplitude__mean', 'amplitude__median', 
                           'amplitude__minimum', 'amplitude__maximum',
                           'magnitude__mean', 'magnitude__median', 
                           'magnitude__minimum', 'magnitude__maximum']
        
        merged_df = merged_df[columns_ordered]
        
        return merged_df
    
    def save_merged_csv(self):
        """
        Saves the merged DataFrame to the output CSV file.
        """
        merged_df = self.merge_csv_files()
        if merged_df is not None:
            merged_df.to_csv(self.output_file, index=False)
            print(f"Merging completed. Results saved to: {self.output_file}")
        else:
            print("Error merging CSV files. No output file was created.")


