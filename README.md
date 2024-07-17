# python_advanced_predictive_maintenance
# Machine Learning Framework für Zeitreihen-Daten
Projektbeschreibung
Dieses Projekt umfasst die Entwicklung eines Python-basierten Machine Learning Frameworks zur Verarbeitung von Zeitreihen-Daten. Das Framework soll verschiedene Machine Learning Modelle trainieren, evaluieren und optimieren können. Der Fokus liegt auf der Erkennung von anormalen Zuständen in Maschinenkomponenten durch Vibrationssensoren, um durch KI-basierte Zustandsüberwachungssysteme einen Wettbewerbsvorteil zu erzielen.

# Kursinformationen
Kurs: Fortgeschrittene Python Programmierung, Sommersemester 2024
Dozent: Christian Seidler
Institution: HS AlbSig

# Teammitglieder und Codeverantwortung
-Dominique Saile (max.mustermann@example.com) - Datenlade-Modul
-Christina Maria Richard (erika.mustermann@example.com) - Datenvorverarbeitungs-Modul
-Anshel Nohl (nohlansh@hs-albsig.de) - Dataloader, Pipeline, Pytorch-learner, Pytorch-evaluator 
-Niklas Bukowski (jane.smith@example.com) - Modellimplementierungs-Modul
-Luca-David Stegmaier (alex.mueller@example.com) - Modelltraining und Evaluations-Modul

# Beschreibung des Datensatzes
Der verwendete Datensatz ist der "Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection" (MIMII), erstellt von Hitachi im Jahr 2019. Er enthält Aufnahmen von vier Maschinentypen: Ventile, Pumpen, Ventilatoren und Gleitschienen, sowohl im Normal- als auch im Anomaliebetrieb. Jede Aufnahme ist eine 10-sekündige Zeitreihe mit einer Abtastrate von 16 kHz, mit hinzugefügtem Fabriklärm bei verschiedenen Signal-Rausch-Verhältnissen (SNRs).

Link zum Datensatz: https://zenodo.org/records/3384388
Link zum Paper: https://arxiv.org/pdf/1909.09347 

# Schritte zur Ausführung der Skripte
Repository von GitHub klonen.
Umgebung requirements.txt einrichten. Über den command pip install -r requirements.txt

Skripte mit den bereitgestellten Konfigurationsdateien konfigurieren.
In ihr wird ein ein Parameter-Grid, das verschiedene Hyperparameter-Kombinationen spezifiziert. Hier ist eine Erklärung der einzelnen Parameter:

epochs: Anzahl der Epochen, über die das Modell trainiert werden soll.
hidden_dim1, hidden_dim2, hidden_dim3: Dimensionen der ersten und zweiten versteckten Schicht im Netzwerk.
batch_size: Größe der Batches für das Training.
lr: Lernrate für die Optimierung des Modells.
optimizer: Typ des Optimierungsalgorithmus (Momentan implementiertsind allerdings nur Adam, SGD und RMSprop).
dropout_percentage: Dropout-Raten für die Regularisierung des Netzwerks.


# Manuelle Merkmalsextraktion 
Modelle: ANN, RandomForest, DecisionTree
Ergebnisse: 

# Zeitreihe als Input
Modelle: [Liste der Modelle]
Ergebnisse: [Zusammenfassung der Ergebnisse]

# Spektrogramm als Input
Modelle: Neuronales Netz
Ergebnisse: 
Accuracy: 99,13%
F1 Score: 97,03%
Confusion Matrix:
[[195   1]
 [  1  33]]
