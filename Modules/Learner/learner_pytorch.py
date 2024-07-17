"""
Filename: learner_pytorch.py
Author:Anshel Nohl <nohalansh@hs-albsig.de>

Created at: 2024-06-29
Last changed: 2024-07-14
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from typing import List, Tuple

# ChatGPT (ID: 03), ChatGPT (ID: 04), ChatGPT (ID: 05)
class SpectrogramDataset(Dataset):
    def __init__(self, file_list: List[str]) -> None:
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
        label = 0 if 'abnormal' in file_name else 1  
        data = np.load(file_name)
        
        scaler = StandardScaler()
        data_scaled = torch.from_numpy(scaler.fit_transform(data)).float()
        
        return data_scaled, label

class SpectrogramNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int = 128, hidden_dim2: int = 64, hidden_dim3: int = 32, dropout_percentage: float = 0.5) -> None:
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
        self.dropout1 = nn.Dropout(dropout_percentage)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_percentage)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.dropout3 = nn.Dropout(dropout_percentage)
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
