import os
import pandas as pd

from Modules.Dataloader.data_intake import DataLoader
from Modules.FeatureExtractor.featureextractor_amplitude import FeatureExtractor
from Modules.FeatureExtractor.featureextractor_magnitude import FeatureExtractor
from Modules.FeatureExtractor.featureextractor_spectogram import FeatureExtractor
from Modules.FeatureExtractor.merge_CSV import CSVMerger
from Modules.Evaluator.Evaluator_common import Evaluator
from Modules.Learner.learner import Learner


def main():
    # Definiere die Pfade
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    wav_directory = r"D:\0_dB_pump\pump\id_00\abnormal"
    csv_folder_path = os.path.join(base_path, 'CSV')
    
    # 1. Verarbeite WAV-Dateien und speichere die Ergebnisse in CSV-Dateien
    dataloader = DataLoader(wav_directory)
    
    print("Verarbeite Amplitude Daten...")
    amplitude_df = dataloader.verarbeite_wav_dateien(funktion="amplitude", save_plots=True)
    amplitude_csv_path = os.path.join(csv_folder_path, "amplituden_dfs.csv")
    amplitude_df.to_csv(amplitude_csv_path, index=False)
    print(f"Amplitude Daten gespeichert unter {amplitude_csv_path}")

    print("Verarbeite Magnitude Daten...")
    magnitude_df = dataloader.verarbeite_wav_dateien(funktion="magnitude", save_plots=True)
    magnitude_csv_path = os.path.join(csv_folder_path, "magnituden_dfs.csv")
    magnitude_df.to_csv(magnitude_csv_path, index=False)
    print(f"Magnitude Daten gespeichert unter {magnitude_csv_path}")

    print("Verarbeite Spectrogram Daten...")
    spectrogram_df = dataloader.verarbeite_wav_dateien(funktion="spectrogram", save_plots=True)
    spectrogram_csv_path = os.path.join(csv_folder_path, "spectrogram_dfs.csv")
    spectrogram_df.to_csv(spectrogram_csv_path, index=False)
    print(f"Spectrogram Daten gespeichert unter {spectrogram_csv_path}")

    # 2. Extrahiere Merkmale aus den Amplitude CSV-Daten
    print("Extrahiere Merkmale aus den Amplitude Daten...")
    feature_extractor = FeatureExtractor(amplitude_csv_path)
    feature_extractor.process_data()

    # 3. Extrahiere Merkmale aus den Amplitude CSV-Daten
    print("Extrahiere Merkmale aus den Magnituden Daten...")
    feature_extractor = FeatureExtractor(magnitude_csv_path)
    feature_extractor.process_data()

    # 4. Extrahiere Merkmale aus den Amplitude CSV-Daten
    print("Extrahiere Merkmale aus den Magnituden Daten...")
    feature_extractor = FeatureExtractor(spectrogram_csv_path)
    feature_extractor.process_data()


    # 5. Füge die CSV-Dateien zusammen
    print("Füge die CSV-Dateien zusammen...")
    feature_csv_folder_path = os.path.join(base_path, 'CSV_Features')
    amplitude_features_csv_path = os.path.join(feature_csv_folder_path, "extracted_features_amplitude.csv")
    magnitude_features_csv_path = os.path.join(feature_csv_folder_path, "extracted_features_magnitude.csv")
    spectrogram_features_csv_path = os.path.join(feature_csv_folder_path, "extracted_features_spectrogram.csv")

    merger = CSVMerger(amplitude_features_csv_path, magnitude_features_csv_path, spectrogram_features_csv_path)
    merger.save_merged_csv()

    learner = Learner()
    learner.run_learner()

if __name__ == '__main__':
    main()
