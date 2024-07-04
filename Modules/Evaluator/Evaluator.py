import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve,auc
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
import os
from Modules.Learner.learner import Learner  # Ensure correct import path

class Evaluator:
    def __init__(self, learner):
        self.learner = learner
        # Determine current script directory
        current_script_path = os.path.dirname(os.path.abspath(__file__))

        # Navigate two levels up from current script to 'Gruppe3' folder
        base_folder_path = os.path.join(current_script_path, '..', '..')

        # Construct path to 'CSV_Features' folder
        self.features_folder_path = os.path.join(base_folder_path, 'CSV_Features')

        # Path where models are saved
        self.model_folder_path = os.path.join(base_folder_path, "Model_Storage")

    def compare_models(self):
        """
        Compares the accuracies of all models and outputs the results.
        """
        results = pd.DataFrame({"Model": ["ANN", "DecisionTree"], "Score": [self.learner.ANN_accuracy, self.learner.D_tree_accuracy]})
        results = results.sort_values(by="Score", ascending=False)
        results = results.set_index("Score")
        print(results)

    def confusion_matrices(self):
        """
        Plots the confusion matrices for all models.
        """
        model_names = ["ANN", "DecisionTree"]

        for model_name in model_names:
            if model_name == "ANN":
                y_pred = self.learner.predict_ANN
                y_pred = np.where(y_pred > 0.5, 1, 0)
            elif model_name == "DecisionTree":
                y_pred = self.learner.predict_decision_tree
            else:
                print("Invalid model name")
                continue

            cm = confusion_matrix(self.learner.y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.learner.y_test))
            disp.plot()
            plt.title(f"Confusion Matrix for {model_name}")
            plt.show()

    def roc_curves(self):
        """
        Plots the ROC Curves for all models.
        """
        # Liste erstellen mit allen Modellnamen
        model_names = ["ANN", "DecisionTree"]

        # Iteriere über alle Modellnamen mit den jeweiligen Vorhersagen
        for model_name in model_names:
            if model_name == "ANN":
                y_pred = self.learner.predict_ANN
            elif model_name == "DecisionTree":
                 y_pred = self.learner.predict_decision_tree
            else:
                print("Invalid model name")
                continue

            # Initialisieren und Plotten der ROC Curves
            fpr, tpr, thresholds = roc_curve(self.learner.y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {model_name}')
            plt.legend(loc="lower right")
            plt.show()



    def load_our_models(self):
        model_ANN = load_model(f"{self.model_folder_path}/ANN_model.h5")
        self.learner.predict_ANN = model_ANN.predict(self.learner.X_test)
        model_DecisionTree = joblib.load(f"{self.model_folder_path}/Decision_Tree.pkl")
        self.learner.predict_decision_tree = model_DecisionTree.predict(self.learner.X_test)

if __name__ == "__main__":
    # Instantiate the Learner class
    learner_instance = Learner()
    learner_instance.run_learner()  # This will train your models and set predictions

    # Instantiate the Evaluator class with the Learner instance
    evaluator_instance = Evaluator(learner_instance)

    # Execute methods of the Evaluator class
    evaluator_instance.compare_models()
    evaluator_instance.confusion_matrices
    evaluator_instance.roc_curves()
    evaluator_instance.load_our_models()
