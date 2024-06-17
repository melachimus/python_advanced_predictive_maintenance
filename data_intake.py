import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
from typing import List

def create_amplitude_df(file_path: str, resampled_data: np.ndarray, sample_rate: int, channels: List[int], save_dir: str = None) -> pd.DataFrame:
    """
    Creates a DataFrame for the temporal amplitude variation and saves visualizations.

    Parameters:
    -----------
    file_path : str
        Path to the WAV file.
    resampled_data : np.ndarray
        Resampled data for all channels.
    sample_rate : int
        Sample rate of the resampled data.
    channels : list of int
        List of channels to process.
    save_dir : str, optional
        Directory to save the plot.

    Returns:
    ------------
    pd.DataFrame
        DataFrame containing the amplitude over time.
    """
    file_name = os.path.basename(file_path)
    amplitude_list = []
    for channel in channels:
        channel_data = resampled_data[:, channel].astype(np.float32)
        channel_data /= np.max(np.abs(channel_data))
        duration = len(channel_data) / sample_rate
        time = np.linspace(0., duration, len(channel_data))
        amplitude_df = pd.DataFrame({
            'file_name': [file_name] * len(time),
            'channel': [channel] * len(time),
            'time': time,
            'amplitude': channel_data
        })
        amplitude_list.append(amplitude_df)
        
        if save_dir:
            plt.figure(figsize=(10, 4))
            plt.plot(time, channel_data)
            plt.title(f'Temporal Amplitude Variation (Channel {channel})')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude')
            plt.grid(True)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_name = f"{os.path.splitext(file_name)[0]}_channel_{channel}.png"
            plt.savefig(os.path.join(save_dir, plot_name))
            plt.close()
    
    return pd.concat(amplitude_list, ignore_index=True)

def create_magnitude_df(file_path: str, resampled_data: np.ndarray, sample_rate: int, channels: List[int], save_dir: str = None) -> pd.DataFrame:
    """
    Creates a DataFrame for the magnitude spectrum and saves visualizations.

    Parameters:
    -----------
    file_path : str
        Path to the WAV file.
    resampled_data : np.ndarray
        Resampled data for all channels.
    sample_rate : int
        Sample rate of the resampled data.
    channels : list of int
        List of channels to process.
    save_dir : str, optional
        Directory to save the plot.

    Returns:
    ------------
    pd.DataFrame
        DataFrame containing the magnitude spectrum of the different frequencies.
    """
    file_name = os.path.basename(file_path)
    magnitude_list = []
    for channel in channels:
        channel_data = resampled_data[:, channel].astype(np.float32)
        channel_data /= np.max(np.abs(channel_data))
        freqs = np.fft.rfftfreq(len(channel_data), 1/sample_rate)
        magnitudes = np.abs(np.fft.rfft(channel_data))
        magnitude_df = pd.DataFrame({
            'file_name': [file_name] * len(freqs),
            'channel': [channel] * len(freqs),
            'frequency': freqs,
            'magnitude': magnitudes
        })
        magnitude_list.append(magnitude_df)

        if save_dir:
            plt.figure(figsize=(10, 4))
            plt.plot(freqs, magnitudes)
            plt.title(f'Magnitude Spectrum (Channel {channel})')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            plt.grid(True)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_name = f"{os.path.splitext(file_name)[0]}_channel_{channel}.png"
            plt.savefig(os.path.join(save_dir, plot_name))
            plt.close()
    
    return pd.concat(magnitude_list, ignore_index=True)

def create_spectrogram_df(file_path: str, resampled_data: np.ndarray, sample_rate: int, channels: List[int], save_dir: str = None) -> pd.DataFrame:
    """
    Creates a DataFrame for the spectrogram and saves visualizations.

    Parameters:
    -----------
    file_path : str
        Path to the WAV file.
    resampled_data : np.ndarray
        Resampled data for all channels.
    sample_rate : int
        Sample rate of the resampled data.
    channels : list of int
        List of channels to process.
    save_dir : str, optional
        Directory to save the plot.

    Returns:
    ------------
    pd.DataFrame
        DataFrame containing the spectrogram.
    """
    file_name = os.path.basename(file_path)
    spectrogram_list = []
    for channel in channels:
        channel_data = resampled_data[:, channel].astype(np.float32)
        channel_data /= np.max(np.abs(channel_data))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(channel_data)), ref=np.max)
        times = librosa.times_like(D, sr=sample_rate)
        frequencies = librosa.fft_frequencies(sr=sample_rate)
        spectrogram_df = pd.DataFrame(D, index=frequencies, columns=times)
        spectrogram_df = spectrogram_df.stack().reset_index()
        spectrogram_df.columns = ['frequency', 'time', 'amplitude']
        spectrogram_df['file_name'] = file_name
        spectrogram_df['channel'] = channel
        spectrogram_df = spectrogram_df[["file_name", "channel", "frequency", "time", "amplitude"]]
        spectrogram_list.append(spectrogram_df)
      
        if save_dir:
            plt.figure(figsize=(10, 4))
            D_plot = spectrogram_df.pivot(index='frequency', columns='time', values='amplitude')
            img = librosa.display.specshow(D_plot.values, sr=sample_rate, x_axis='time', y_axis='log')
            plt.colorbar(img, format='%+2.0f dB')
            plt.title(f'Spectrogram (Channel {channel})')
            plt.xlabel('Time [s]')
            plt.ylabel('Frequency [Hz]')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plot_name = f"{os.path.splitext(file_name)[0]}_channel_{channel}.png"
            plt.savefig(os.path.join(save_dir, plot_name))
            plt.close()
    
    return pd.concat(spectrogram_list, ignore_index=True)

def verarbeite_wav_dateien(verzeichnis: str, funktion: str, target_sample_rate: int = 16000, channels: List[int] = [0,1,2,3,4,5,6,7], save_plots: bool = False, plots_dir: str = None) -> pd.DataFrame:
    """
    Processes multiple .wav files in a directory according to the desired function and returns the result in a DataFrame.
    Optionally saves visualizations.

    Args:
        verzeichnis (str): Directory containing the .wav files.
        funktion (str): Type of data to extract from the .wav files ('amplitude', 'magnitude', 'spectrogram').
        target_sample_rate (int): Target sample rate for the .wav files (default is 16000 Hz).
        channels (list): List of channels to use.
        save_plots (bool): Whether to save visualizations (default is False).
        plots_dir (str): Directory to save visualizations (required if save_plots is True).

    Returns:
        pd.DataFrame: DataFrame with the temporal amplitude, magnitude spectrum, or spectrogram data.
    """
    valide_funktionen = ["amplitude", "magnitude", "spectrogram"]
    if funktion not in valide_funktionen:
        raise ValueError(f"Ungültige Option '{funktion}'. Erlaubte Optionen sind: {', '.join(valide_funktionen)}")
    
    alle_dfs = []
    for root, dirs, files in os.walk(verzeichnis):
        for file in files:
            if file.endswith('.wav'):
                dateipfad = os.path.join(root, file)
                print(f"Verarbeite Datei: {dateipfad}")
                ordnername = os.path.basename(os.path.dirname(dateipfad))
                label = 'abnormal' if 'abnormal' in ordnername.lower() else 'normal'

                data, _ = librosa.load(dateipfad, sr=target_sample_rate)
                
                if plots_dir:
                    save_dir = plots_dir + "_" + label 
                else:
                    save_dir = plots_dir
                if funktion == "amplitude":
                    df = create_amplitude_df(dateipfad, data, target_sample_rate, channels, save_dir=save_dir)
                elif funktion == "magnitude":
                    df = create_magnitude_df(dateipfad, data, target_sample_rate, channels, save_dir=save_dir)
                elif funktion == "spectrogram":
                    df = create_spectrogram_df(dateipfad, data, target_sample_rate, channels, save_dir=save_dir)
                
                df['Label'] = label
                alle_dfs.append(df)
    
    return pd.concat(alle_dfs, ignore_index=True)

# Example usage:
# amplituden_dfs = verarbeite_wav_dateien(r"C:\Users\anohl\OneDrive\Dokumente\A_Uni_stuff\Albstadt\Semester 1\Python advanced\Prüfungsleitung\python_advanced_predictive_maintenance\raw_data", funktion="amplitude", target_sample_rate=1000, channels=[0])
# amplituden_dfs.to_csv("amplituden_dfs.csv")
# magnitude_dfs = verarbeite_wav_dateien(r"C:\Users\anohl\OneDrive\Dokumente\A_Uni_stuff\Albstadt\Semester 1\Python advanced\Prüfungsleitung\python_advanced_predictive_maintenance\raw_data", funktion="magnitude", target_sample_rate=1000, channels=[0])
# magnitude_dfs.to_csv("magnituden_dfs.csv")
spectrogram_dfs = verarbeite_wav_dateien(r"C:\Users\anohl\OneDrive\Dokumente\A_Uni_stuff\Albstadt\Semester 1\Python advanced\Prüfungsleitung\python_advanced_predictive_maintenance\raw_data", funktion="spectrogram", target_sample_rate=1000, channels=[0], save_plots=True, plots_dir="spectrogram_pics")
spectrogram_dfs.to_csv("spectrogram_dfs.csv")

