"""
Filename: pipeline.py
Author:Luca-David Stegmaier <stegmalu@hs-albsig.de>, Niklas Bukowski <bukowsni@hs-albsig.de>, Dominique Saile <sailedom@hs-albsig.de>, Christina Maria Richard <richarch@hs-albsig.de>, Anshel Nohl <nohalansh@hs-albsig.de>

Created at: 2024-06-24
Last changed: 2024-07-16
"""
import os
from Modules.Evaluator.evaluator_pytorch import run_tuning_pipeline
import pandas as pd
import json
from Modules.Dataloader.data_intake import DataLoader
from Modules.FeatureExtractor.featureextractor_amplitude import FeatureExtractor as AmplitudeFeatureExtractor
from Modules.FeatureExtractor.featureextractor_magnitude import FeatureExtractor as MagnitudeFeatureExtractor
# # from Modules.FeatureExtractor.featureextractor_spectogram import FeatureExtractor as SpectrogramFeatureExtractor
from Modules.FeatureExtractor.merge_csv2 import CSVMerger
from Modules.Evaluator.Evaluator_common import Evaluator
from Modules.Learner.learner import Learner

with open("config.json", "r") as file:
    config = json.load(file)

pytorch_param_grid = config["param_grid_pytorch"]

BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR,"raw_data")
FILE_NAME_PYTORCH = config["file_name_pytorch"]

def main():
    dataloader = DataLoader(verzeichnis=DATA_PATH, target_sample_rate=1000)

    # Amplitude Daten verarbeiten und speichern
    print("Verarbeite Amplitude Daten...")
    amplituden_dfs = dataloader.verarbeite_wav_dateien(funktion="amplitude", save_plots=True)
    amplituden_dfs.to_csv(os.path.join(dataloader.csv_folder_path, "amplituden_dfs.csv"))
    print("Amplitude Daten gespeichert.")

    # Magnitude Daten verarbeiten und speichern
    print("Verarbeite Magnitude Daten...")
    magnitude_dfs = dataloader.verarbeite_wav_dateien(funktion="magnitude", save_plots=True)
    magnitude_dfs.to_csv(os.path.join(dataloader.csv_folder_path, "magnituden_dfs.csv"))
    print("Magnitude Daten gespeichert.")

    # Spectrogram Daten verarbeiten und speichern
    print("Verarbeite Spectrogram Daten...")
    spectrogram_dfs = dataloader.verarbeite_wav_dateien(funktion="spectrogram", save_plots=True)
    spectrogram_dfs.to_csv(os.path.join(dataloader.csv_folder_path, "spectrogram_dfs.csv"))
    print("Spectrogram Daten gespeichert.")

    amplitude_input_file = os.path.join(dataloader.csv_folder_path, "amplituden_dfs.csv")
    magnitude_input_file = os.path.join(dataloader.csv_folder_path, "magnituden_dfs.csv")
 

    amplitude_extractor = AmplitudeFeatureExtractor(amplitude_input_file)
    amplitude_extractor.process_data()

    magnitude_extractor = MagnitudeFeatureExtractor(magnitude_input_file)
    magnitude_extractor.process_data()

    # CSV Merging
    amplitude_features_file = amplitude_extractor.output_file
    magnitude_features_file = magnitude_extractor.output_file

    csv_merger = CSVMerger(amplitude_features_file, magnitude_features_file)
    csv_merger.save_merged_csv()

    # Model Training der manuellen Merkmalsextraktion

    merged_csv_path = os.path.join(BASE_DIR, "CSV_Features", "Merge_CSV.csv")
    learner = Learner(merged_csv_path)
    learner.run_learner()

    # Model Evaluation der manuellen Merkmalsextraktion
    evaluator = Evaluator(learner)
    evaluator.compare_models()
    evaluator.confusion_matrices()
    evaluator.roc_curves()
    evaluator.load_our_models()

    # Model evaluation und training Pytorch
    run_tuning_pipeline(base_directory=BASE_DIR, param_grid=pytorch_param_grid, filename=FILE_NAME_PYTORCH)

if __name__ == '__main__':
    main()
