"""
Filename: pytorch.py
Author:Anshel Nohl <nohalansh@hs-albsig.de>

Created at: 2024-06-29
Last changed: 2024-07-06
"""


import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any

class SpectrogramDataset(Dataset):
    def __init__(self, file_list: list[str]) -> None:
        """
        Initialisiert das Dataset mit einer Liste von Dateinamen.
        
        Args:
            file_list (List[str]): Liste der Dateinamen.
        """
        self.file_list = file_list
        
    def __len__(self) -> int:
        """
        Gibt die Anzahl der Elemente im Dataset zurück.
        
        Returns:
            int: Anzahl der Elemente.
        """
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Gibt das Datenelement und das Label für einen gegebenen Index zurück.
        
        Args:
            idx (int): Index des Datenelements.
        
        Returns:
            Tuple[torch.Tensor, int]: Datenelement und zugehöriges Label.
        """
        file_name = self.file_list[idx]
        label = 1 if 'abnormal' in file_name else 0  
        data = np.load(file_name)
        
        scaler = StandardScaler()
        data_scaled = torch.from_numpy(scaler.fit_transform(data)).float()
        
        return data_scaled, label

class SpectrogramNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64, hidden_dim3: int = 32) -> None:
        """
        Initialisiert das neuronale Netz mit den angegebenen Dimensionen.
        
        Args:
            input_dim (int): Dimension des Eingangs.
            hidden_dim1 (int): Dimension der ersten versteckten Schicht. Standardwert ist 128.
            hidden_dim2 (int): Dimension der zweiten versteckten Schicht. Standardwert ist 64.
            hidden_dim3 (int): Dimension der dritten versteckten Schicht. Standardwert ist 32.
        """
        super(SpectrogramNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(hidden_dim3, 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Definiert den Vorwärtsdurchlauf des Netzwerks.
        
        Args:
            x (torch.Tensor): Eingabedaten.
        
        Returns:
            torch.Tensor: Ausgabedaten.
        """
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x

class SpectrogramClassifier:
    def __init__(self, train_files: list[str], test_files: list[str], input_dim: int, params: dict[str, int|str]) -> None:
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

    def evaluate_model(self) -> float:
        """
        Bewertet das Modell basierend auf den Testdaten.
        
        Returns:
            float: F1-Score des Modells auf den Testdaten.
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

        return f1

    def save_model(self, model_path: str) -> None:
        """
        Speichert das Modell unter dem angegebenen Pfad.
        
        Args:
            model_path (str): Pfad zum Speichern des Modells.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

def tune_hyperparameters(train_files: List[str], test_files: list[str], input_dim: int, param_grid: list[dict[str, int|str]]) -> None:
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

    for params in param_grid:
        print(f'Training with parameters: {params}')
        classifier = SpectrogramClassifier(train_files, test_files, input_dim, params)
        classifier.preprocess_data()
        classifier.initialize_model()
        classifier.train_model()
        f1 = classifier.evaluate_model()

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_classifier = classifier

    if best_classifier is not None:
        model_path = os.path.join(os.getcwd(), "Models", "best_model.pth")
        best_classifier.save_model(model_path)
        print(f"Best model had the follwoing params: {best_params}")
        print(f'Best model saved at {model_path} with F1 Score: {best_f1}')

# Beispiel für die Nutzung des Frameworks
def run_tuning_pipeline(base_directory: str, param_grid: dict[str, int|str]) -> None:

    data_directory = os.path.join(base_directory, "Bilder_Daten")
    file_list = [os.path.join(data_directory, f) for f in os.listdir(data_directory)]

    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)
    train_dataset = SpectrogramDataset(train_files)
    sample_train_data, sample_train_label = train_dataset[0]

    input_dim = sample_train_data.numel()

    param_grid = list(ParameterGrid(param_grid))
    tune_hyperparameters(train_files, test_files, input_dim, param_grid)