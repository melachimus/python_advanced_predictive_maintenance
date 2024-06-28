import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

class DataLoader:
    def __init__(self, verzeichnis, target_sample_rate=16000):
        self.verzeichnis = verzeichnis
        self.target_sample_rate = target_sample_rate

        # Ermitteln des Verzeichnisses des aktuellen Skripts
        current_script_path = os.path.dirname(os.path.abspath(__file__))

        # Zwei Ebenen über dem aktuellen Skript zum 'Gruppe7'-Ordner wechseln
        base_folder_path = os.path.join(current_script_path, '..', '..')

        # Den Pfad zum Graphen-Ordner konstruieren
        self.graph_folder_path = os.path.join(base_folder_path, 'Graphen')
        if not os.path.exists(self.graph_folder_path):
            os.makedirs(self.graph_folder_path)

        # Den Pfad zum CSV-Ordner konstruieren
        self.csv_folder_path = os.path.join(base_folder_path, 'CSV')
        if not os.path.exists(self.csv_folder_path):
            os.makedirs(self.csv_folder_path)

    def create_amplitude_df(self, file_path, resampled_data, sample_rate, save_plots):
        file_name = os.path.basename(file_path)
        resampled_data /= np.max(np.abs(resampled_data))
        duration = len(resampled_data) / sample_rate
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
            plot_name = f"{os.path.splitext(file_name)[0]}_amplitude.png"
            plt.savefig(os.path.join(self.graph_folder_path, plot_name))
            plt.close()

        return amplitude_df

    def create_magnitude_df(self, file_path, resampled_data, sample_rate, save_plots):
        file_name = os.path.basename(file_path)
        resampled_data /= np.max(np.abs(resampled_data))
        freqs = np.fft.rfftfreq(len(resampled_data), 1/sample_rate)
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
            plot_name = f"{os.path.splitext(file_name)[0]}_magnitude.png"
            plt.savefig(os.path.join(self.graph_folder_path, plot_name))
            plt.close()

        return magnitude_df

    def create_spectrogram_df(self, file_path, resampled_data, sample_rate, save_plots):
        file_name = os.path.basename(file_path)
        resampled_data /= np.max(np.abs(resampled_data))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(resampled_data)), ref=np.max)
        times = librosa.times_like(D, sr=sample_rate)
        frequencies = librosa.fft_frequencies(sr=sample_rate)
        spectrogram_df = pd.DataFrame(D, index=frequencies, columns=times)
        spectrogram_df = spectrogram_df.stack().reset_index()
        spectrogram_df.columns = ['frequency', 'time', 'amplitude']
        spectrogram_df['file_name'] = file_name
        spectrogram_df = spectrogram_df[["file_name", "frequency", "time", "amplitude"]]

        if save_plots:
            plt.figure(figsize=(10, 4))
            D_plot = spectrogram_df.pivot(index='frequency', columns='time', values='amplitude')
            img = librosa.display.specshow(D_plot.values, sr=sample_rate, x_axis='time', y_axis='log')
            plt.colorbar(img, format='%+2.0f dB')
            plt.title(f'Spectrogram')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            plot_name = f"{os.path.splitext(file_name)[0]}_spectrogram.png"
            plt.savefig(os.path.join(self.graph_folder_path, plot_name))
            plt.close()

        return spectrogram_df

    def verarbeite_wav_dateien(self, funktion, save_plots=False):
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

                    data, _ = librosa.load(dateipfad, sr=self.target_sample_rate)
                    if funktion == "amplitude":
                        df = self.create_amplitude_df(dateipfad, data, self.target_sample_rate, save_plots)
                    elif funktion == "magnitude":
                        df = self.create_magnitude_df(dateipfad, data, self.target_sample_rate, save_plots)
                    elif funktion == "spectrogram":
                        df = self.create_spectrogram_df(dateipfad, data, self.target_sample_rate, save_plots)

                    df['Label'] = label
                    alle_dfs.append(df)

        return pd.concat(alle_dfs, ignore_index=True)

# Beispielhafte Verwendung:
dataloader = DataLoader(r"D:\0_dB_pump\pump\id_00\abnormal")

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

