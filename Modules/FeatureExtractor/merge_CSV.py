import pandas as pd
import os

class CSVMerger:
    def __init__(self, file1, file2, file3):
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.create_output_file()  # Aufruf der Methode zur Erstellung des Ausgabepfads
    
    def create_output_file(self):
        # Definiere den Pfad zum Speichern der CSV-Datei
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_script_path, '..', '..'))
        csv_storage_path = os.path.join(base_path, 'CSV_Features')
        os.makedirs(csv_storage_path, exist_ok=True)
        self.output_file = os.path.join(csv_storage_path, 'Merge_CSV.csv')

    def merge_csv_files(self):
        # Einlesen der CSV-Dateien
        try:
            df1 = pd.read_csv(self.file1)
            df2 = pd.read_csv(self.file2)
            df3 = pd.read_csv(self.file3)
        except FileNotFoundError as e:
            print(f"Fehler beim Lesen der CSV-Dateien: {e}")
            return None
        
        # Sortieren der DataFrames nach Schritt und Label
        df1.sort_values(by=['Schritt', 'Label'], inplace=True)
        df2.sort_values(by=['Schritt', 'Label'], inplace=True)
        df3.sort_values(by=['Schritt', 'Label'], inplace=True)
        
        # Merge der DataFrames nach Schritt und Label
        merged_df = pd.merge(df1, df2, on=['Schritt', 'Label'])
        merged_df = pd.merge(merged_df, df3, on=['Schritt', 'Label'])
        
        # Sortieren der Spalten in der gewünschten Reihenfolge
        columns_ordered = ['Schritt',
                           'amplitude__mean', 'amplitude__median', 'amplitude__variance', 'amplitude__standard_deviation',
                           'amplitude__minimum', 'amplitude__maximum',
                           'magnitude__mean', 'magnitude__median', 'magnitude__variance', 'magnitude__standard_deviation',
                           'magnitude__minimum', 'magnitude__maximum',
                           'frequency__mean', 'frequency__median', 'frequency__variance', 'frequency__standard_deviation',
                           'frequency__minimum', 'frequency__maximum',
                           'Label']
        
        merged_df = merged_df[columns_ordered]
        
        return merged_df
    
    def save_merged_csv(self):
        merged_df = self.merge_csv_files()
        if merged_df is not None:
            merged_df.to_csv(self.output_file, index=False)
            print(f"Merge abgeschlossen. Ergebnisse gespeichert in: {self.output_file}")
        else:
            print("Fehler beim Merge der CSV-Dateien. Keine Ausgabedatei erstellt.")

# Beispiel-Nutzung der Klasse
if __name__ == '__main__':
    file1 = r''
    file2 = r''
    file3 = r''

    merger = CSVMerger(file1, file2, file3)
    merger.save_merged_csv()
