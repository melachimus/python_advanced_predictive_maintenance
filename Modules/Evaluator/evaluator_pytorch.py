"""
Filename: evaluator_pytorch.py
Author:Anshel Nohl <nohalansh@hs-albsig.de>

Created at: 2024-06-29
Last changed: 2024-07-16
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split, ParameterGrid
from Modules.Learner.learner_pytorch import SpectrogramNet, SpectrogramDataset
import matplotlib.pyplot as plt
import seaborn as sns

# ChatGPT (ID: 03), ChatGPT (ID: 04), ChatGPT (ID: 05)
class SpectrogramClassifier:
    def __init__(self, train_files: List[str], test_files: List[str], input_dim: int, params: Dict[str, Any]) -> None:
        """
        Initialisiert den Klassifikator mit den gegebenen Parametern.
        
        Args:
            train_files (List[str]): Liste der Trainingsdateien.
            test_files (List[str]): Liste der Testdateien.
            input_dim (int): Dimension des Eingangs.
            params (Dict[str, Any]): Hyperparameter für den Klassifikator.
        """
        self.train_files = train_files
        self.test_files = test_files
        self.input_dim = input_dim
        self.params = params

        self.num_epochs = params.get("epochs", 20)
        self.batch_size = params.get('batch_size', 64)
        self.lr = params.get('lr', 0.001)
        self.optimizer_name = params.get('optimizer', 'adam')
        self.loss_function_name = params.get('loss_function', 'cross_entropy')
        self.dropout_percentage = params.get("dropout_percentage", 0.5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.test_loader = None

    def preprocess_data(self) -> None:
        """
        Bereitet die Daten für das Training und Testen vor.
        """
        train_dataset = SpectrogramDataset(self.train_files)
        test_dataset = SpectrogramDataset(self.test_files)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def initialize_model(self) -> None:
        """
        Initialisiert das Modell mit den gegebenen Parametern.
        """
        hidden_dim1 = self.params.get('hidden_dim1', 128)
        hidden_dim2 = self.params.get('hidden_dim2', 64)
        hidden_dim3 = self.params.get('hidden_dim3', 32)
        droput_percentage = self.params.get("dropout_percentage", 0.5)
        
        self.model = SpectrogramNet(self.input_dim, hidden_dim1, hidden_dim2, hidden_dim3, droput_percentage).to(self.device)

    def get_optimizer(self) -> optim.Optimizer:
        """
        Gibt den Optimierer basierend auf den angegebenen Parametern zurück.
        
        Returns:
            optim.Optimizer: Der Optimizer.
        """
        if self.optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def train_model(self) -> None:
        """
        Trainiert das Modell basierend auf den gegebenen Daten und Hyperparametern.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = self.get_optimizer()

        for epoch in range(self.num_epochs):
            self.model.train()
            for batch_data, batch_labels in self.train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_data.to(self.device))
                loss = criterion(outputs, batch_labels.to(self.device))
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {round(loss.item(), 4)}')

    def evaluate_model(self) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Bewertet das Modell anhand der Testdaten.

        Returns:
            tuple: Ein Tuple mit F1-Score, Genauigkeit, Confusion Matrix, FPR, TPR und ROC-AUC.
        """
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for batch_data, batch_labels in self.test_loader:
                outputs = self.model(batch_data.to(self.device))
                probs = nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy()[:, 1])  

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        print(f'Accuracy on test set: {round(accuracy, 2)}')
        print(f'F1 Score on test set: {round(f1, 2)}')
        print(f'Confusion Matrix on test set:\n{cm}')

        return f1, accuracy, cm, fpr, tpr, roc_auc

    def save_model(self, model_path: str) -> None:
        """
        Speichert das trainierte Modell.

        Args:
            model_path (str): Der Pfad, unter dem das Modell gespeichert wird.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

def tune_hyperparameters(base_directory: str, train_files: List[str], test_files: List[str], input_dim: int, param_grid: List[Dict[str, Any]], filename: str) -> None:
    """
    Optimiert die Hyperparameter des Modells und speichert das beste Modell.
    Außerdem werden die Confusion Matrix und die ROC-Kurve geplottet und ebenfalls gespeichert.

    Args:
        base_directory (str): Basisverzeichnis, in dem die Modellinformationen gespeichert werden.
        train_files (List[str]): Liste der Trainingsdateien.
        test_files (List[str]): Liste der Testdateien.
        input_dim (int): Dimension des Eingangs.
        param_grid (List[Dict[str, Any]]): Liste der Hyperparameterkombinationen.
        filename (str): Dateiname für das gespeicherte Modell und die Ergebnisse.
    """
    best_f1 = 0.0
    best_params = None
    best_classifier = None
    best_accuracy = 0.0
    best_cm = None

    for params in param_grid:
        print(f'Training with parameters: {params}')
        classifier = SpectrogramClassifier(train_files, test_files, input_dim, params)
        classifier.preprocess_data()
        classifier.initialize_model()
        classifier.train_model()
        f1, accuracy, cm, fpr, tpr, roc_auc = classifier.evaluate_model()

        if f1 > best_f1:
            best_f1 = f1
            best_accuracy = accuracy
            best_cm = cm
            best_fpr = fpr
            best_tpr = tpr
            best_roc_auc = roc_auc
            best_params = params
            best_classifier = classifier

    if best_classifier is not None:
        model_directory = os.path.join(base_directory, "Model_Storage")
        os.makedirs(model_directory, exist_ok=True)
        
        model_path = os.path.join(model_directory, f"{filename}_model.pth")
        best_classifier.save_model(model_path)
        
        result_file_path = os.path.join(model_directory, f"{filename}_model_info.txt")
        with open(result_file_path, "w") as file:
            file.write(f"Best {filename} model had the following params: {best_params}\n")
            file.write(f'Best {filename} model saved at {model_path}\n')
            file.write(f"F1 Score {filename}: {best_f1}\n")
            file.write(f"Accuracy {filename}: {best_accuracy}\n")
            file.write(f"Confusion Matrix {filename}:\n{best_cm}\n")
        
        # Plot der ROC Curve für das beste Pytorch Model
        plt.figure()
        plt.title(f"ROC curve für das beste {filename} Model")
        plt.plot(best_fpr, best_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {round(best_roc_auc, 2)})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(model_directory, f"{filename}_ROC_curve.png"))
        plt.show()
        

        # Plot der Confusion Matrix für das beste Pytorch Model
        plt.figure()
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix {filename}')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.xticks([0.5, 1.5], ['Class 0', 'Class 1'])
        plt.yticks([0.5, 1.5], ['Class 0', 'Class 1'])
        plt.savefig(os.path.join(model_directory, f"{filename}_confusion_matrix.png"))
        plt.show()
        
        print(f'Best model saved at {model_path} with F1 Score: {best_f1}')
        print(f'Model details written to {result_file_path}')

# ChatGPT (ID: 06)
def run_tuning_pipeline(base_directory: str, param_grid: Dict[str, List[Any]], filename: str) -> None:
    """
    Führt eine Hyperparameter-Tuning-Pipeline zum Training eines Modells auf Spektrogramm-Datensätzen durch.

    Diese Funktion organisiert den Prozess des Ladens von Daten, das Aufteilen in Trainings- und Testsets
    und das Abstimmen der Hyperparameter des Modells basierend auf dem angegebenen Parameter-Raster.

    Args:
        base_directory (str): Das Basisverzeichnis, das das Unterverzeichnis "Bilder_Daten" mit den Datendateien enthält.
        param_grid (Dict[str, List[Any]]):  Ein Wörterbuch, das das Raster der zu optimierenden Hyperparameter definiert.
                                            Schlüssel repräsentieren Hyperparameternamen, und Werte sind Listen möglicher Werte.
        filename (str): Der Dateiname für das gespeicherte Modell und die Ergebnisse.

    Das Parameter-Raster sollte die folgenden Hyperparameter enthalten:
    - 'hidden_dim1': Liste der Ganzzahlen für die Dimensionen der ersten versteckten Schicht.
    - 'hidden_dim2': Liste der Ganzzahlen für die Dimensionen der zweiten versteckten Schicht.
    - 'hidden_dim3': Liste der Ganzzahlen für die Dimensionen der dritten versteckten Schicht.
    - 'batch_size': Liste der Ganzzahlen für Batch-Größen.
    - 'lr': Liste der Fließkommazahlen für Lernraten.
    - 'optimizer': Liste der Zeichenketten für Optimierertypen (z.B. 'adam', 'sgd', 'rmsprop').
    - 'dropout_percentage': Liste von floats der dropout chances.

    Returns:
        None

    Diese Funktion gibt keinen Wert zurück, sondern bereitet das hyperparamter-tuning vor.
    """
    data_directory = os.path.join(base_directory, "Bilder_Daten")
    file_list = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]

    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)
    train_dataset = SpectrogramDataset(train_files)
    sample_train_data, sample_train_label = train_dataset[0]

    input_dim = sample_train_data.numel()

    param_grid = list(ParameterGrid(param_grid))
    tune_hyperparameters(base_directory, train_files, test_files, input_dim, param_grid, filename)
