import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
import configparser
from Modules.Learner.learner import learner
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
import joblib
import os

class Evaluator:
        def __init__(self, learner):
            
            self.learner = learner
            # Ermitteln des Verzeichnisses des aktuellen Skripts
            current_script_path = os.path.dirname(os.path.abspath(__file__))

            # Zwei Ebenen über dem aktuellen Skript zum 'Gruppe3'-Ordner wechseln
            base_folder_path = os.path.join(current_script_path, '..', '..')

            # Den Pfad zum 'Datenset'-Ordner konstruieren
            self.features_folder_path = os.path.join(base_folder_path, 'CSV_Features')

            # Den Pfad wo die modelle gespeichert werden
            self.model_folder_path = os.path.join(base_folder_path, "Model_Storage")
        def compare_models(self):
            """
            Vergleicht die Genauigkeiten aller Modelle und gibt die Ergebnisse aus.
            """
            # Dataframe wird erstellt aus den jeweiligen Modellnamen und deren dazugehörigen Accuracy Scores
            results = pd.DataFrame({"Model":["ANN","DecisionTree"],"Score":[self.learner.ANN_accuracy,self.learner.D_tree_accuracy]})
            results = results.sort_values(by="Score", ascending=False)
            results = results.set_index("Score")
            print(results)

        def confusion_matrices(self):
            """
            Plottet die Confusion Matrices für alle Modelle.
            """
            # Liste erstellen mit allen Modellnamen
            model_names = ["ANN", "DecisionTree"]

         # Iteriere über alle Modellnamen mit den jeweiligen Vorhersagen
            for model_name in model_names:
                if model_name == "ANN":
                    y_pred = self.learner.predict_ANN
                    y_pred = np.where(y_pred > 0.5, 1, 0)
                elif model_name == "DecisionTree":
                    y_pred = self.learner.predict_decision_tree

                else:
                    print("Invalid model name")
                    continue

                # Initialisieren und Plotten der Confusion Matrizen
                cm = confusion_matrix(self.learner.y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.learner.y_test))
                disp.plot()
                plt.title(f"Confusion Matrix for {model_name}")
                plt.show()


        def load_our_models(self):
            model_ANN = load_model(f"{self.model_folder_path}/{'ANN_model.h5'}")
            self.learner.predict_ANN = model_ANN.predict()
            model_DecisionTree = joblib.load(f"{self.model_folder_path}/{'Decision_Tree.pkl'}")
            self.learner.predict_decision_tree = model_DecisionTree.predict()