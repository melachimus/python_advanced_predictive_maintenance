import pandas as pd
import os

class CSVMerger:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2
        self.create_output_file()  
    
    def create_output_file(self):
        # Definiere den Pfad zur Speicherung der CSV-Datei
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_script_path, '..', '..'))
        csv_storage_path = os.path.join(base_path, 'CSV_Features')
        os.makedirs(csv_storage_path, exist_ok=True)
        self.output_file = os.path.join(csv_storage_path, 'Merge_CSV.csv')

    def merge_csv_files(self):
        # Lade die gesamten Eingabedaten
        try:
            df1 = pd.read_csv(self.file1)
            df2 = pd.read_csv(self.file2)
        except FileNotFoundError as e:
            print(f"Fehler beim Lesen der CSV-Dateien: {e}")
            return None
        
        # Sortiere die DataFrames nach 'id' und 'Label'
        df1.sort_values(by=['id', 'Label'], inplace=True)
        df2.sort_values(by=['id', 'Label'], inplace=True)

        # Merge die DataFrames auf 'id' und 'Label'
        merged_df = pd.merge(df1, df2, on=['id', 'Label'])

        # Sortiere die Spalten in der gewünschten Reihenfolge
        columns_ordered = ['id', 'Label', 
                           'amplitude__mean', 'amplitude__median', 
                           'amplitude__minimum', 'amplitude__maximum',
                           'magnitude__mean', 'magnitude__median', 
                           'magnitude__minimum', 'magnitude__maximum',
                           ]
        
        merged_df = merged_df[columns_ordered]
        
        return merged_df
    
    def save_merged_csv(self):
        merged_df = self.merge_csv_files()
        if merged_df is not None:
            merged_df.to_csv(self.output_file, index=False)
            print(f"Mergen abgeschlossen. Ergebnisse gespeichert unter: {self.output_file}")
        else:
            print("Fehler beim Mergen der CSV-Dateien. Es wurde keine Ausgabedatei erstellt.")


