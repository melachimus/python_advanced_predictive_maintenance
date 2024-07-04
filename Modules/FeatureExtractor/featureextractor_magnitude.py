import os
import pandas as pd
from tsfresh import extract_features

class FeatureExtractor:
    def __init__(self, input_file):
        self.input_file = input_file
        self.create_output_file()  # Direkt beim Initialisieren die Ausgabedatei erstellen
        
    def clean_data(self, data):
        if 'magnitude' in data.columns:
            magnitude_data = data[['magnitude']]
        else:
            magnitude_data = pd.DataFrame()
        
        magnitude_data = magnitude_data.fillna(magnitude_data.mean())
        
        # Füge eine Spalte 'id' hinzu, um die Zeilen zu nummerieren
        magnitude_data['id'] = range(len(magnitude_data))
        
        return magnitude_data

    def create_output_file(self):
        # Define path to save CSV file
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_script_path, '..', '..'))
        csv_storage_path = os.path.join(base_path, 'CSV_Features')
        os.makedirs(csv_storage_path, exist_ok=True)
        self.output_file = os.path.join(csv_storage_path, 'extracted_features_magnitude.csv')

        # Erstelle eine leere DataFrame mit den Spaltenbezeichnungen und speichere sie in die CSV-Datei
        initial_data = {
            'id': [],
            'Label': [],
            'file_name': [],
            'magnitude__mean': [],
            'magnitude__median': [],
            'magnitude__minimum': [],
            'magnitude__maximum': []
        }
        initial_df = pd.DataFrame(initial_data)
        initial_df.to_csv(self.output_file, index=False)

        print(f"Leere CSV-Datei erstellt: {self.output_file}")

    def process_data(self):
        # Lese die gesamten Eingabedaten ein
        try:
            data = pd.read_csv(self.input_file)
        except Exception as e:
            print(f"Fehler beim Lesen der Eingabedatei: {str(e)}")
            return

        if 'Label' not in data.columns:
            print("Keine 'Label'-Spalte in den Eingabedaten gefunden.")
            return

        # Gruppiere nach 'file_name' und 'Label'
        grouped_data = data.groupby(['file_name', 'Label'])

        current_id = 0  # Startwert für die id-Zuweisung

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
                        column_id='id',  # Verwende 'id' als column_id für tsfresh
                        column_value='magnitude',
                        default_fc_parameters=custom_fc_parameters
                    )

                    # Aggregate die extrahierten Merkmale
                    aggregated_features = extracted_features.mean().to_frame().T  # Mean über alle Spalten

                    # Füge 'file_name' und 'Label' hinzu
                    aggregated_features['file_name'] = file_name
                    aggregated_features['Label'] = label

                    # Füge eine 'id' Spalte hinzu und durchnummeriere die Zeilen
                    aggregated_features['id'] = current_id
                    current_id += 1  # Inkrementiere die id für die nächste Gruppe

                    # Reihenfolge der Spalten anpassen, um 'id' und 'Label' zuerst zu haben
                    columns_ordered = ['id', 'Label', 'file_name', 'magnitude__mean', 'magnitude__median',
                                       'magnitude__minimum',
                                       'magnitude__maximum']
                    aggregated_features = aggregated_features[columns_ordered]

                    # Speichere die aggregierten Merkmale und das Label in die CSV-Datei
                    aggregated_features.to_csv(self.output_file, mode='a', header=False, index=False)

                    # Nachricht für jeden Gruppenabschluss ausgeben
                    print(f"Verarbeitet file_name: {file_name}, Label: {label}")

                except Exception as e:
                    print(f"Fehler beim Verarbeiten der Daten für file_name: {file_name}, Label: {label}: {str(e)}")

            else:
                print(f"In den Daten für file_name: {file_name}, Label: {label} wurde keine 'magnitude'-Spalte gefunden.")

        print("Feature-Extraktion abgeschlossen und in", self.output_file, "gespeichert")

if __name__ == '__main__':
    input_file = r'C:\Users\CR\python_advanced_predictive_maintenance\CSV\magnituden_dfs.csv'

    extractor = FeatureExtractor(input_file)
    extractor.process_data()
