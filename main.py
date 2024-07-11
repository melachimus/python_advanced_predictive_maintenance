import os
import pandas as pd

from Modules.Dataloader.data_intake import DataLoader
from Modules.FeatureExtractor.featureextractor_amplitude import FeatureExtractor as AmplitudeFeatureExtractor
from Modules.FeatureExtractor.featureextractor_magnitude import FeatureExtractor as MagnitudeFeatureExtractor
from Modules.FeatureExtractor.featureextractor_spectogram import FeatureExtractor as SpectrogramFeatureExtractor
from Modules.FeatureExtractor.merge_csv2 import CSVMerger
from Modules.Evaluator.Evaluator_common import Evaluator
from Modules.Learner.learner import Learner

def main():
    # Step 1: Data Loading and Preprocessing
    dataloader = DataLoader(r"D:\0_dB_pump\pump\id_00")

    # Amplitude Data Processing and Saving
    print("Verarbeite Amplitude Daten...")
    amplituden_dfs = dataloader.verarbeite_wav_dateien(funktion="amplitude", save_plots=True)
    amplituden_dfs.to_csv(os.path.join(dataloader.csv_folder_path, "amplituden_dfs.csv"))
    print("Amplitude Daten gespeichert.")

    # Magnitude Data Processing and Saving
    #print("Verarbeite Magnitude Daten...")
    magnitude_dfs = dataloader.verarbeite_wav_dateien(funktion="magnitude", save_plots=True)
    magnitude_dfs.to_csv(os.path.join(dataloader.csv_folder_path, "magnituden_dfs.csv"))
    print("Magnitude Daten gespeichert.")


    # Step 2: Feature Extraction
    amplitude_input_file = os.path.join(dataloader.csv_folder_path, "amplituden_dfs.csv")
    magnitude_input_file = os.path.join(dataloader.csv_folder_path, "magnituden_dfs.csv")

    amplitude_extractor = AmplitudeFeatureExtractor(amplitude_input_file)
    amplitude_extractor.process_data()

    magnitude_extractor = MagnitudeFeatureExtractor(magnitude_input_file)
    magnitude_extractor.process_data()

    # Step 3: CSV Merging
    amplitude_features_file = amplitude_extractor.output_file
    magnitude_features_file = magnitude_extractor.output_file

    csv_merger = CSVMerger(amplitude_features_file, magnitude_features_file)
    csv_merger.save_merged_csv()

    # Step 4: Model Training and Evaluation
    merged_csv_path = csv_merger.output_file
    learner = Learner(merged_csv_path)
    learner.run_learner()

    # Step 5: Model Evaluation
    evaluator = Evaluator(learner)
    evaluator.compare_models()
    evaluator.confusion_matrices()
    evaluator.roc_curves()
    evaluator.load_our_models()

if __name__ == "__main__":
    main()