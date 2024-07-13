"""
Filename: pipeline.py
Author:Luca-David Stegmaier <stegmalu@hs-albsig.de>, Niklas Bukowski <bukowsni@hs-albsig.de>, Dominique Saile <sailedom@hs-albsig.de>, Christina Maria Richard <richarch@hs-albsig.de>, Anshel Nohl <nohalansh@hs-albsig.de>

Created at: 2024-06-24
Last changed: 2024-07-06
"""
import os
import pytorch_script
import data_intake

BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR,"raw_data")
dataloader = data_intake.DataLoader(DATA_PATH, target_sample_rate=1000)

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

param_grid = {
    'hidden_dim1': [128, 256],
    'hidden_dim2': [64, 128],
    'hidden_dim3': [32, 64],
    'batch_size': [32, 64],
    'lr': [0.001, 0.005],
    'optimizer': ['adam', 'sgd', 'rmsprop'],
}

pytorch_script.run_tuning_pipeline(base_directory=BASE_DIR, param_grid=param_grid)
