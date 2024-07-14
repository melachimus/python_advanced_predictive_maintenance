"""
Filename: evaluator_pytorch.py
Author:Anshel Nohl <nohalansh@hs-albsig.de>

Created at: 2024-06-29
Last changed: 2024-07-12
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split, ParameterGrid
from Modules.Learner.learner_pytorch import SpectrogramNet, SpectrogramDataset

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

        self.batch_size = params.get('batch_size', 64)
        self.num_epochs = 20
        self.lr = params.get('lr', 0.001)
        self.optimizer_name = params.get('optimizer', 'adam')
        self.loss_function_name = params.get('loss_function', 'cross_entropy')

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
        
        self.model = SpectrogramNet(self.input_dim, hidden_dim1, hidden_dim2, hidden_dim3).to(self.device)

    def get_optimizer(self) -> optim.Optimizer:
        """
        Gibt den Optimierer basierend auf den angegebenen Parametern zurück.
        
        Returns:
            optim.Optimizer: Der Optimierer.
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

    def evaluate_model(self) -> Tuple[float, float, np.ndarray]:
        """
        Bewertet das Modell basierend auf den Testdaten.
        
        Returns:
            Tuple[float, float, np.ndarray]: F1-Score, Accuracy und Confusion Matrix des Modells auf den Testdaten.
        """
        self.model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch_data, batch_labels in self.test_loader:
                outputs = self.model(batch_data.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)

        print(f'Accuracy on test set: {round(accuracy, 2)}')
        print(f'F1 Score on test set: {round(f1, 2)}')
        print(f'Confusion Matrix on test set:\n{cm}')

        return f1, accuracy, cm

    def save_model(self, model_path: str) -> None:
        """
        Speichert das Modell unter dem angegebenen Pfad.
        
        Args:
            model_path (str): Pfad zum Speichern des Modells.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

def tune_hyperparameters(base_directory: str, train_files: List[str], test_files: List[str], input_dim: int, param_grid: List[Dict[str, Any]]) -> None:
    """
    Optimiert die Hyperparameter des Modells und speichert das beste Modell.
    
    Args:
        train_files (List[str]): Liste der Trainingsdateien.
        test_files (List[str]): Liste der Testdateien.
        input_dim (int): Dimension des Eingangs.
        param_grid (List[Dict[str, Any]]): Liste der Hyperparameterkombinationen.
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
        f1, accuracy, cm = classifier.evaluate_model()

        if f1 > best_f1:
            best_f1 = f1
            best_accuracy = accuracy
            best_cm = cm
            best_params = params
            best_classifier = classifier

    if best_classifier is not None:
        model_directory = os.path.join(base_directory, "Model_Storage")
        os.makedirs(model_directory, exist_ok=True)
        
        model_path = os.path.join(model_directory, "best_pytorch_model.pth")
        best_classifier.save_model(model_path)
        
        result_file_path = os.path.join(model_directory, "best_pytorch_model_info.txt")
        with open(result_file_path, "w") as file:
            file.write(f"Best Pytorch model had the following params: {best_params}\n")
            file.write(f'Best Pytorch model saved at {model_path}\n')
            file.write(f"F1 Score Pytorch: {best_f1}\n")
            file.write(f"Accuracy Pytorch: {best_accuracy}\n")
            file.write(f"Confusion Matrix Pytorch:\n{best_cm}\n")
        
        print(f'Best model saved at {model_path} with F1 Score: {best_f1}')
        print(f'Model details written to {result_file_path}')

def run_tuning_pipeline(base_directory: str, param_grid: Dict[str, List[Any]]) -> None:
    """
    Run a hyperparameter tuning pipeline for training a model on spectrogram datasets.

    This function orchestrates the process of loading data, splitting it into training 
    and testing sets, and then tuning the model's hyperparameters based on the provided
    parameter grid.

    Parameters:
    base_directory (str): The base directory containing the "Bilder_Daten" subdirectory with the data files.
    param_grid (Dict[str, List[Any]]): A dictionary defining the grid of hyperparameters to be tuned. 
                                     Keys represent hyperparameter names, and values are lists of possible values.

    The parameter grid should include the following hyperparameters:
    - 'hidden_dim1': List of integers for the first hidden layer dimensions.
    - 'hidden_dim2': List of integers for the second hidden layer dimensions.
    - 'hidden_dim3': List of integers for the third hidden layer dimensions.
    - 'batch_size': List of integers for batch sizes.
    - 'lr': List of floats for learning rates.
    - 'optimizer': List of strings for optimizer types (e.g., 'adam', 'sgd', 'rmsprop').

    Returns:
    None

    This function does not return any value but will print out the results of the hyperparameter tuning.
    """
    data_directory = os.path.join(base_directory, "Bilder_Daten")
    file_list = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]

    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)
    train_dataset = SpectrogramDataset(train_files)
    sample_train_data, sample_train_label = train_dataset[0]

    input_dim = sample_train_data.numel()

    param_grid = list(ParameterGrid(param_grid))
    tune_hyperparameters(base_directory, train_files, test_files, input_dim, param_grid)
