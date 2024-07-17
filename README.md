# python_advanced_predictive_maintenance
# Machine Learning Framework für Zeitreihen-Daten
Projektbeschreibung
Dieses Projekt umfasst die Entwicklung eines Python-basierten Machine Learning Frameworks zur Verarbeitung von Zeitreihen-Daten. Das Framework soll verschiedene Machine Learning Modelle trainieren, evaluieren und optimieren können. Der Fokus liegt auf der Erkennung von anormalen Zuständen in Maschinenkomponenten durch Vibrationssensoren, um durch KI-basierte Zustandsüberwachungssysteme einen Wettbewerbsvorteil zu erzielen.

# Kursinformationen
Kurs: Fortgeschrittene Python Programmierung, Sommersemester 2024
Dozent: Christian Seidler
Institution: HS AlbSig

# Teammitglieder und Codeverantwortung
-Dominique Saile (sailedom@hs-albsig.de) - Manuelle Merkmalsextraktion (ANN, DecisionTree, RandomForest), Pipeline
-Christina Maria Richard (richarch@hs-albsig.de) - Manuelle Merkmalsextraktion (ANN, Decision Tree, Random Forest), Pipeline
-Anshel Nohl (nohlansh@hs-albsig.de) - Dataloader, Pipeline, Pytorch-learner, Pytorch-evaluator 
-Niklas Bukowski (bukowsni@hs-albsig.de) - Zeitreihen, Rockets.py (Rocket & MiniRocket)
-Luca-David Stegmaier (stegmalu@hs-albsig.de) - Zeitreihen, (Inception Time)

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
Accuracy Score für folgende Modelle:
ANN: 84,78%
RandomForest: 90,87%
DecisionTree: 93,48%

# Zeitreihe als Input
Modelle ; Ergebnisse: 
"Rocket_BA_0.74_2407171407.pkl" ; Balanced Accuracy: 73,6% - F1-Score: 91,6%
"MiniRocket_BA_0.72_2407171307.pkl" ; Balanced Accuracy: 71,9% - F1-Score: 91%
"InceptionTime.pkl" ; Balanced Accuracy: 96,55% - F1-Score: 98,69% - Precision: 98,69%

# Spektrogramm als Input
Modelle: Neuronales Netz
Ergebnisse: 
Accuracy: 98,70%
F1 Score: 99,24%
Confusion Matrix:

[[ 31   3]​
 [  0 196]]

# Lessons Learned
Wir haben gelernt, dass es in so einem Projekt essenziell ist, von Anfang an eine klare Kommunikation innerhalb der Gruppe bezüglich der Ordnerstruktur und Herangehensweise sicherzustellen. 
Ebenso wichtig ist es, sich auf einheitliche Benennungsrichtlinien zu einigen, damit alle Teammitglieder denselben Namenskonventionen folgen. Dies fördert eine effizientere Zusammenarbeit und verhindert Missverständnisse, Verwirrung oder Konflikte während der Entwicklung.
Es ist ebenfalls entscheidend, die Teammitglieder über Änderungen bestehende oder zukünftige Änderungen im Code oder der Herangehensweise zu informieren um eine bessere Planung zu ermöglichen und Merge-Konflikte zu vermeiden.
