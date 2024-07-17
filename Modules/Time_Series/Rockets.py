"""
Filename: Rockets.py
Author: Niklas Bukowski <bukowsni@hs-albsig.de>

Version: Relative Paths, Classes, Run Directory

Created at: 2024-07-13
Last changed: 2024-07-16
"""

import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tsai.all import *
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from typing import Optional, Tuple

class RunManager:
    def __init__(self, base_dir: str, dataset_name: str, classifier_type: str):
        """
        Erstellt eine neue Instanz der Klasse RunManager und richtet das Verzeichnis für den Run ein.

        Args:
        - base_dir: Basisverzeichnis, in dem das run-Verzeichnis erstellt wird.
        - dataset_name: Name des verwendeten Datensatzes.
        - classifier_type: Typ des Klassifikators, der verwendet wird.
        """
        self.base_dir = os.path.join(base_dir, "rocket_runs")
        os.makedirs(self.base_dir, exist_ok=True)
        self.run_dir = os.path.join(self.base_dir, self.create_run_folder_name(dataset_name, classifier_type))
        os.makedirs(self.run_dir, exist_ok=True)

    def create_run_folder_name(self, dataset_name: str, classifier_type: str) -> str:
        """
        Erstellt einen eindeutigen Ordnernamen für den Run basierend auf dem Datensatz und dem Klassifikatortyp.

        Args:
        - dataset_name: Name des verwendeten Datensatzes.
        - classifier_type: Typ des Klassifikators, der verwendet wird.

        Returns:
        - Ein eindeutiger Ordnername für den Run.
        """
        timestamp = datetime.now().strftime("%y%m%d%H%m")
        return f"{timestamp}_{dataset_name}_{classifier_type}"

    def get_run_path(self) -> str:
        """
        Gibt den Pfad des aktuellen Run Directory zurück.
        """
        return self.run_dir
    
    def get_run_folder_name(self) -> str:
        """
        Gibt den Namen des aktuellen Run Directory zurück.
        """
        return os.path.basename(self.run_dir)
    
class DataHandler:
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialisiert eine neue Instanz der Klasse DataHandler.

        Args:
        - base_dir: Basisverzeichnis, in dem die Daten gespeichert sind.
        """
        self.base_dir = base_dir

    def load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Lädt Daten aus einer CSV-Datei.

        Args:
        - file_path: Relativer Pfad zur CSV-Datei.

        Returns:
        - Ein Pandas DataFrame mit den geladenen Daten.
        """
        full_path = os.path.join(self.base_dir, file_path)
        try:
            print("Loading data from file...")
            df = pd.read_csv(full_path)
            print("Data loaded.")
            return df
        except FileNotFoundError:
            print(f"Error: The file {full_path} was not found.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file {full_path} is empty.")
        except pd.errors.ParserError:
            print(f"Error: The file {full_path} could not be parsed.")
        except Exception as e:
            print(f"An unexpected error occurred while loading the data: {e}")
            return None

    def prep_data(self, df: pd.DataFrame, sample_col: str, feat_col: str, target_col: str, data_cols: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Bereitet die Daten für das Training und Testen vor.

        Args:
        - df: Das Pandas DataFrame, das die Daten enthält.
        - sample_col: Name der Spalte, die die Stichprobenkennungen enthält.
        - feat_col: Name der Spalte, die die Merkmale enthält.
        - target_col: Name der Spalte, die die Zielwerte enthält.
        - data_cols: Liste der Spalten, die für die Modellierung verwendet werden sollen.

        Returns:
        - X_train, y_train, X_test, y_test: Arrays der Trainings- und Testdaten.
        """
        X, y = df2xy(df, sample_col=sample_col, feat_col=feat_col, target_col=target_col, data_cols=data_cols, y_func=lambda o: o)
        print(f'X shape: {X.shape}, y shape: {y.shape}')
        splits = get_splits(y, valid_size=.2, stratify=True, random_state=42, shuffle=True, show_plot=False)
        X_train, y_train = X[splits[0]], y[splits[0]]
        X_test, y_test = X[splits[1]], y[splits[1]]
        print("Data preparation complete.")
        return X_train, y_train, X_test, y_test

class ModelHandler:
    def __init__(self, classifier_type: str = 'Rocket', random_state: int = 42, run_manager: Optional[RunManager] = None):
        """
        Initialisiert eine neue Instanz der Klasse ModelHandler.

        Args:
        - classifier_type: Typ des Klassifikators, der verwendet wird (z.B. 'Rocket', 'MiniRocket').
        - random_state: Zufallsstatus für die Reproduzierbarkeit.
        - run_manager: Instanz der Klasse RunManager für den aktuellen Lauf.
        """
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.model = None
        self.run_manager = run_manager

    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, model_file_name: Optional[str] = None) -> np.ndarray:
        """
        Trainiert das Modell und gibt Vorhersagen zurück.

        Args:
        - X_train: Trainingsdaten.
        - y_train: Zielwerte für die Trainingsdaten.
        - X_test: Testdaten.
        - model_file_name: Name der Datei eines gespeicherten Modells (optional).

        Returns:
        - y_pred: Vorhersagen für die Testdaten.
        """
        if model_file_name:
            self.model = self.load_model(model_file_name, self.run_manager.base_dir)
        else:
            if self.classifier_type.lower() == 'rocket':
                self.model = RocketClassifier(random_state=self.random_state)
            elif self.classifier_type.lower() == 'minirocket':
                self.model = MiniRocketClassifier(random_state=self.random_state)
            else:
                raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
            self.model.fit(X_train, y_train)
            print(f"{self.classifier_type} classifier fitted.")
        
        y_pred = self.model.predict(X_test)
        print("Prediction made.")
        return y_pred

    def save_model(self, balanced_acc: float) -> None:
        """
        Speichert das trainierte Modell mit Referenz zur Modelgüte im Dateinamen.

        Args:
        - balanced_acc: Balanced Accuracy des Modells.
        """
        try:
            timestamp = datetime.now().strftime("%y%m%d%H%m")
            filename = f"{self.classifier_type}_BA_{balanced_acc:.2f}_{timestamp}.pkl"
            filepath = os.path.join(self.run_manager.get_run_path(), filename)
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to Run Directory.")
        except PermissionError:
            print(f"Error: Permission denied while trying to save the model to {filepath}.")
        except Exception as e:
            print(f"An unexpected error occurred while saving the model: {e}")

    def load_model(self, model_file_name: str, base_dir: str) -> Optional:
        """
        Lädt ein gespeichertes Modell.

        Args:
        - model_file_name: Name der Datei des gespeicherten Modells.
        - base_dir: Basisverzeichnis, in dem das Modell gespeichert ist.

        Returns:
        - Das geladene Modell.
        """
        try:
            for root, dirs, files in os.walk(base_dir):
                if model_file_name in files:
                    filepath = os.path.join(root, model_file_name)
                    with open(filepath, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"Model loaded from {filepath}")
                    return self.model
            raise FileNotFoundError(f"File {model_file_name} not found in {base_dir}")
        except FileNotFoundError as e:
            print(e)
        except pickle.UnpicklingError:
            print(f"Error: The file {model_file_name} could not be unpickled.")
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")
            return None

class Evaluation:
    def __init__(self, y_test: np.ndarray, y_pred: np.ndarray, X_test: np.ndarray, run_manager: Optional[RunManager] = None):
        """
        Initialisiert eine neue Instanz der Klasse Evaluation und führt diese aus.

        Args:
        - y_test: Wahre Zielwerte für die Testdaten.
        - y_pred: Vorhergesagte Zielwerte.
        - X_test: Testdaten.
        - run_manager: Instanz der Klasse RunManager für den aktuellen Lauf.
        """
        self.y_test = y_test
        self.y_pred = y_pred
        self.X_test = X_test
        self.balanced_acc = None
        self.metrics_report = ""
        self.run_manager = run_manager

        self.calculate_metrics()
        self.save_metrics()
        self.generate_confusion_matrix()
        self.show_results("Correctly_Classified_Instances")
        self.show_results("Misclassified_Instances")
        print("Evaluation saved in Run Directory.")

    def calculate_metrics(self) -> None:
        """
        Berechnet die Leistungskennzahlen des Modells.

        Args:
        - None

        Returns:
        - None
        """
        self.balanced_acc = balanced_accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')

        report = classification_report(self.y_test, self.y_pred)

        self.metrics_report = (
            f'Balanced Accuracy: {self.balanced_acc}\n'
            f'F1-Score: {f1}\n'
            f'Precision: {precision}\n'
            f'Recall: {recall}\n\n'
            f'Classification Report:\n{report}'
        )

    def save_metrics(self) -> None:
        """
        Speichert die berechneten Leistungskennzahlen in einer Textdatei.

        Args:
        - None

        Returns:
        - None
        """
        try:
            run_folder_name = self.run_manager.get_run_folder_name()
            title = f"{run_folder_name}\n\n"
            full_report = title + self.metrics_report

            filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = os.path.join(self.run_manager.get_run_path(), filename)
            with open(filepath, 'w') as f:
                f.write(full_report)
        except PermissionError:
            print(f"Error: Permission denied while trying to save the metrics to {filepath}.")
        except Exception as e:
            print(f"An unexpected error occurred while saving the metrics: {e}")

    def generate_confusion_matrix(self) -> None:
        """
        Generiert und speichert die Konfusionsmatrix als Bild.

        Args:
        - None

        Returns:
        - None
        """
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test), ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        self.save_plot(fig, "Confusion_Matrix")

    def show_results(self, title: str) -> None:
        """
        Zeigt und speichert die Ergebnisse der korrekt und inkorrekt klassifizierten Instanzen.

        Args:
        - title: Titel für die angezeigten Ergebnisse.

        Returns:
        - None
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        correct_indices = np.where(self.y_test == self.y_pred)[0]
        incorrect_indices = np.where(self.y_test != self.y_pred)[0]
        indices = correct_indices if 'Correct' in title else incorrect_indices

        for i, idx in enumerate(indices[:4]):
            ax = axs[i // 2, i % 2]
            ax.plot(self.X_test[idx, 0], lw=2)
            ax.set_title(f'True: {self.y_test[idx]} \nPred: {self.y_pred[idx]}', color='green' if self.y_test[idx] == self.y_pred[idx] else 'red')
        
        fig.suptitle(title)
        plt.tight_layout()
        self.save_plot(fig, title)

    def save_plot(self, fig: plt.Figure, title: str, subfolder: str = "") -> None:
        """
        Speichert ein Plot als Bilddatei.

        Args:
        - fig: Die zu speichernde Matplotlib-Figur.
        - title: Titel des Plots.
        - subfolder: Unterverzeichnis zum Speichern des Plots (optional).

        Returns:
        - None
        """
        try:
            path = os.path.join(self.run_manager.get_run_path(), subfolder)
            os.makedirs(path, exist_ok=True)
            timestamp = datetime.now().strftime("%y%m%d%H%m")
            filename = f"{title}_{timestamp}.png"
            filepath = os.path.join(path, filename)
            fig.savefig(filepath)
            plt.close(fig)
        except PermissionError:
            print(f"Error: Permission denied while trying to save the plot to {filepath}.")
        except Exception as e:
            print(f"An unexpected error occurred while saving the plot: {e}")

if __name__ == "__main__":
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_handler = DataHandler(base_dir)

    still_testing = True
    if still_testing == True:
        
        #Select Dataset
        dataset = "MIMII"

        if dataset == "OliveOil":
            # UCR_data OliveOil:
            X_train, y_train, X_test, y_test = get_UCR_data('OliveOil', Xdtype='float64')

        elif dataset == "MIMII":
            # Provide relative path for the data file
            file_path = "CSV\\sample_data.csv"       #"CSV\\transformed_amplituden_dfs.csv"
            df = data_handler.load_data(file_path)
            X_train, y_train, X_test, y_test = data_handler.prep_data(df, sample_col='sample', feat_col='feature', target_col='target', data_cols=df.columns[2:-1])
        
        classifier_type = 'MiniRocket'

    elif still_testing == False:
        config_file = os.path.join(os.path.dirname(base_dir), 'config.json')
        config = read_config(config_file)
        time_series_file = os.path.join(base_dir, config['amplitude_file'].replace('/', os.path.sep))
        time_series = read_time_series(time_series_file)
        transformed_dataset = transform_dataset(time_series, sample='file_name', target='Label', value='amplitude')
        X_train, y_train, X_test, y_test = data_handler.prep_data(transformed_dataset, sample_col='sample', feat_col='feature', target_col='target', data_cols=df.columns[2:-1])

    run_manager = RunManager(base_dir, dataset, classifier_type)
    model_handler = ModelHandler(classifier_type, 42, run_manager)

    model_file_name = None      #Choose None for training a new Model or load one like this "MiniRocket_BA_0.55.pkl"
    
    # Rocket Classifier trainieren und vorhersage treffen
    y_pred = model_handler.fit_predict(X_train, y_train, X_test, model_file_name )
    
    # Calculate and save metrics
    eval = Evaluation(y_test, y_pred, X_test, run_manager)

    # Save Model
    model_handler.save_model(eval.balanced_acc)

    print("Main script execution finished.")