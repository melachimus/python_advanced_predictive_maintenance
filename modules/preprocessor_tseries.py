"""
Filename: preprocessor_teseries.py
Author:Luca-David Stegmaier <stegmalu@hs-albsig.de>

Created at: 2024-07-14
Last changed: 2024-07-15
"""
import pandas as pd
import os
import json
from typing import Optional


def read_config(config_file: str) -> Optional[dict]:
    """
        Liest Konfigurationsparameter aus einer JSON-Datei.

        Args:
        - config_file: Dateipfad zur JSON-Konfigurationsdatei.

        Returns:
        - Ein Python Dictionary mit den gelesenen Konfigurationsparametern oder None, wenn die Datei nicht gefunden wurde.
        """
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
            print(f"Konfigurationsdatei erfolgreich geladen: {config_file}")
            return config
    except FileNotFoundError:
        print(f"Konfigurationsdatei nicht gefunden: {config_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"Fehler beim Lesen der JSON-Konfigurationsdatei: {config_file}. Fehlermeldung: {str(e)}")
        return None


def read_time_series(time_series_file: str) -> Optional[pd.DataFrame]:
    """
        Verarbeitet Zeitreihendaten aus einer CSV-Datei und gibt sie als DataFrame zurück.

        Args:
        - time_series_file: Dateipfad zur Zeitreihendatei.

        Returns:
        - DataFrame mit den verarbeiteten Zeitreihendaten oder None im Fehlerfall.
        """
    try :
        time_series = pd.read_csv(time_series_file)
        print(f"CSV-Datei erfolgreich geladen: {time_series_file}")
        return time_series
    except FileNotFoundError:
        print(f"Datei nicht gefunden: {time_series_file}")
        return None
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        return None


def transform_dataset(original_dataset: pd.DataFrame, 
                      sample: str = 'sample', 
                      feature: str = 'feature', 
                      target: str = 'target', 
                      time: str = 'time', 
                      value: str = 'value') -> pd.DataFrame:
    """
    Transformiert ein gegebenes DataFrame in eine Pivot-Tabelle.

    Args:
        original_dataset (pd.DataFrame): Das ursprüngliche DataFrame, das transformiert werden soll.
        sample (str, optional): Der Name der Spalte, die die Sample-Namen enthält. Standard ist 'file_name'.
        feature (str, optional): Der Name der Spalte, die die Feature-Werte enthält. Standard ist 'feature'.
        target (str, optional): Der Name der Spalte, die die Target-Werte enthält. Standard ist 'Label'.
        time (str, optional): Der Name der Spalte, die die Zeitwerte enthält. Standard ist 'time'.
        value (str, optional): Der Name der Spalte, die die Amplitudenwerte enthält. Standard ist 'amplitude'.

    Returns:
        pd.DataFrame: Das transformierte DataFrame.
        
    """
    original_dataset['unique_sample'] = original_dataset[sample] + '_' + original_dataset[target]
    original_dataset['sample_num'] = original_dataset['unique_sample'].astype('category').cat.codes

    if feature not in original_dataset.columns:
        original_dataset[feature] = 0
    
    transformed_dataset = original_dataset.pivot_table(index=['sample_num', feature], columns=time, values=value, fill_value=0).reset_index()
    transformed_dataset[target] = transformed_dataset['sample_num'].map(original_dataset.drop_duplicates(subset=['sample_num']).set_index('sample_num')[target])

    # Umbenennen der Spaltennamen
    time_columns = list(range(len(transformed_dataset.columns) - len([sample, feature, target])))
    new_columns = ['sample', 'feature'] + time_columns + ['target']
    transformed_dataset.columns = new_columns

    transformed_dataset = transformed_dataset.drop_duplicates()

    return transformed_dataset


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_file = os.path.join(os.path.dirname(current_dir), 'config.json')
    config = read_config(config_file)

    time_series_file = os.path.join(current_dir, config['amplitude_file'].replace('/', os.path.sep))
    time_series = read_time_series(time_series_file)

    transformed_dataset = transform_dataset(time_series, sample='file_name', target='Label', value='amplitude')
    print(transformed_dataset)
