"""
Filename: data_intake.py
Author:Christina Maria Richard <richarch@hs-albsig.de>, Anshel Nohl <nohalansh@hs-albsig.de>

Created at: 2024-06-24
Last changed: 2024-07-06
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display



class DataLoader:
    def __init__(self, verzeichnis: str, target_sample_rate: int = 16000):
        """
        Initialisiert den DataLoader.

        Args:
        - verzeichnis (str): Pfad zum Verzeichnis mit den WAV-Dateien.
        - target_sample_rate (int): Zielabtastrate für die Audiodaten (Standard: 16000).
        """
        self.verzeichnis = verzeichnis
        self.target_sample_rate = target_sample_rate

        # Dafür sorgen, dass die neuen Daten auf der gleichen Heirarchie ebene abgelegt werden wie die Rohdaten
        base_folder_path = os.path.join(verzeichnis, '..' )

        self.picture_data = os.path.join(base_folder_path, 'Bilder_Daten')
        if not os.path.exists(self.picture_data):
            os.makedirs(self.picture_data)


        # Den Pfad zum Graphen-Ordner konstruieren
        self.graph_folder_path = os.path.join(base_folder_path, 'Graphen')
        if not os.path.exists(self.graph_folder_path):
            os.makedirs(self.graph_folder_path)

        # Den Pfad zum CSV-Ordner konstruieren
        self.csv_folder_path = os.path.join(base_folder_path, 'CSV')
        if not os.path.exists(self.csv_folder_path):
            os.makedirs(self.csv_folder_path)

    def create_amplitude_df(self, file_path: str, resampled_data: np.ndarray, save_plots: bool, label: str) -> pd.DataFrame:
        """
        Erstellt ein DataFrame für die Amplitudeninformationen einer WAV-Datei.

        Args:
        - file_path (str): Pfad zur WAV-Datei.
        - resampled_data (np.ndarray): Resamplete Audiodaten.
        - save_plots (bool): Flag, ob der Plot gespeichert werden soll.
        - label (str): label ob es sich um normale oder abnormale daten handelt.

        Returns:
        - pd.DataFrame: DataFrame mit Zeit- und Amplitudeninformationen.
        """
        file_name = os.path.basename(file_path)
        resampled_data /= np.max(np.abs(resampled_data))
        duration = len(resampled_data) / self.target_sample_rate
        time = np.linspace(0., duration, len(resampled_data))
        amplitude_df = pd.DataFrame({
            'file_name': [file_name] * len(time),
            'time': time,
            'amplitude': resampled_data
        })

        if save_plots:
            plt.figure(figsize=(10, 4))
            plt.plot(time, resampled_data)
            plt.title(f'Temporal Amplitude Variation')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plot_name = f"{os.path.splitext(file_name)[0]}_{label}_amplitude.png"
            plot_folder = os.path.join(self.graph_folder_path, "amplitude")
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder, exist_ok=True)
            plt.savefig(os.path.join(plot_folder, plot_name))
            plt.close()

        return amplitude_df

    def create_magnitude_df(self, file_path: str, resampled_data: np.ndarray, save_plots: bool, label: str) -> pd.DataFrame:
        """
        Erstellt ein DataFrame für die Magnitudeninformationen einer WAV-Datei.

        Args:
        - file_path (str): Pfad zur WAV-Datei.
        - resampled_data (np.ndarray): Resamplete Audiodaten.
        - save_plots (bool): Flag, ob der Plot gespeichert werden soll.
        - label (str): label ob es sich um normale oder abnormale daten handelt.

        Returns:
        - pd.DataFrame: DataFrame mit Frequenz- und Magnitudeninformationen.
        """
        file_name = os.path.basename(file_path)
        resampled_data /= np.max(np.abs(resampled_data))
        freqs = np.fft.rfftfreq(len(resampled_data), 1/self.target_sample_rate)
        magnitudes = np.abs(np.fft.rfft(resampled_data))
        magnitude_df = pd.DataFrame({
            'file_name': [file_name] * len(freqs),
            'frequency': freqs,
            'magnitude': magnitudes
        })

        if save_plots:
            plt.figure(figsize=(10, 4))
            plt.plot(freqs, magnitudes)
            plt.title(f'Magnitude Spectrum')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plot_name = f"{os.path.splitext(file_name)[0]}_{label}_magnitude.png"
            plot_folder = os.path.join(self.graph_folder_path, "magnitude")
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder, exist_ok=True)
            plt.savefig(os.path.join(plot_folder, plot_name))
            plt.close()

        return magnitude_df

    def create_spectrogram_df(self, file_path: str, resampled_data: np.ndarray, save_plots: bool, label: str) -> pd.DataFrame:
        """
        Erstellt ein DataFrame für das Spektrogramm einer WAV-Datei.

        Args:
        - file_path (str): Pfad zur WAV-Datei.
        - resampled_data (np.ndarray): Resamplete Audiodaten.
        - save_plots (bool): Flag, ob der Plot gespeichert werden soll.
        - label (str): label ob es sich um normale oder abnormale daten handelt.

        Returns:
        - pd.DataFrame: DataFrame mit Frequenz-, Zeit- und Amplitudeninformationen des Spektrogramms.
        """
        file_name = os.path.basename(file_path)
        resampled_data /= np.max(np.abs(resampled_data))
        D = np.abs(librosa.stft(resampled_data))
        times = librosa.times_like(D, sr=self.target_sample_rate)
        frequencies = librosa.fft_frequencies(sr=self.target_sample_rate)
        spectrogram_df = pd.DataFrame(D, index=frequencies, columns=times)
        spectrogram_df = spectrogram_df.stack().reset_index()
        spectrogram_df.columns = ['frequency', 'time', 'amplitude']
        spectrogram_df['file_name'] = file_name
        spectrogram_df = spectrogram_df[["file_name", "frequency", "time", "amplitude"]]

        np.save(os.path.join(self.picture_data, f"{os.path.splitext(file_name)[0]}_{label}_spectrogram.npy"), D)

        if save_plots:
            plt.figure(figsize=(10, 4))
            D_plot = spectrogram_df.pivot(index='frequency', columns='time', values='amplitude')
            img = librosa.display.specshow(D_plot.values, sr=self.target_sample_rate, x_axis='time', y_axis='log')
            plt.colorbar(img, format='%+2.0f dB')
            plt.title(f'Spectrogram')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plot_name = f"{os.path.splitext(file_name)[0]}_{label}_spectrogram.png"
            plot_folder = os.path.join(self.graph_folder_path, "spectrogram")
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder, exist_ok=True)
            plt.savefig(os.path.join(plot_folder, plot_name))
            plt.close()

        return spectrogram_df

    def verarbeite_wav_dateien(self, funktion: str, save_plots: bool = False) -> pd.DataFrame:
        """
        Verarbeitet alle WAV-Dateien im angegebenen Verzeichnis und erstellt entsprechende Datenframes.

        Args:
        - funktion (str): Art der Verarbeitung ('amplitude', 'magnitude' oder 'spectrogram').
        - save_plots (bool): Flag, ob die erstellten Plots gespeichert werden sollen (Standard: False).

        Returns:
        - pd.DataFrame: Kombinierter DataFrame aller verarbeiteten Dateien mit entsprechenden Metadaten.
        """
        valide_funktionen = ["amplitude", "magnitude", "spectrogram"]
        if funktion not in valide_funktionen:
            raise ValueError(f"Ungültige Option '{funktion}'. Erlaubte Optionen sind: {', '.join(valide_funktionen)}")

        alle_dfs = []
        for root, dirs, files in os.walk(self.verzeichnis):
            for file in files:
                if file.endswith('.wav'):
                    dateipfad = os.path.join(root, file)
                    print(f"Verarbeite Datei: {dateipfad}")
                    ordnername = os.path.basename(os.path.dirname(dateipfad))
                    label = 'abnormal' if 'abnormal' in ordnername.lower() else 'normal'

                    data, sr = librosa.load(dateipfad, sr=self.target_sample_rate)
                    print(f"Geladene Datei: {dateipfad}, Abtastrate: {sr}")

                    if funktion == "amplitude":
                        df = self.create_amplitude_df(dateipfad, data, save_plots, label=label)
                    elif funktion == "magnitude":
                        df = self.create_magnitude_df(dateipfad, data, save_plots, label=label)
                    elif funktion == "spectrogram":
                        df = self.create_spectrogram_df(dateipfad, data, save_plots, label=label)

                    df['Label'] = label
                    alle_dfs.append(df)

        return pd.concat(alle_dfs, ignore_index=True)


