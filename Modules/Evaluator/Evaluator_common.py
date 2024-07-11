import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, accuracy_score
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

        # Load models from Model_Storage directory
        self.models = {}
        self.load_our_models()

    def compare_models(self):
        """
        Compares the accuracies of all loaded models and outputs the results.
        """
        model_accuracies = {}
        for model_name, model in self.models.items():
            if hasattr(model, 'predict'):
                predictions = model.predict(self.learner.X_test)
                y_pred_binary = np.where(predictions > 0.5, 1, 0)  # Convert to binary predictions
                accuracy = accuracy_score(self.learner.y_test, y_pred_binary)
                model_accuracies[model_name] = accuracy
            else:
                print(f"Model '{model_name}' does not have a predict method.")

        # Sort models by accuracy in descending order
        sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
        results = pd.DataFrame(sorted_models, columns=["Model", "Score"])
        print(results)

    def confusion_matrices(self):
        """
        Plots the confusion matrices for all loaded models.
        """
        for model_name, model in self.models.items():
            if hasattr(model, 'predict'):
                predictions = model.predict(self.learner.X_test)
                y_pred_binary = np.where(predictions > 0.5, 1, 0)  # Convert to binary predictions
                cm = confusion_matrix(self.learner.y_test, y_pred_binary)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.learner.y_test))
                disp.plot()
                plt.title(f"Confusion Matrix for {model_name}")
                plt.show()
            else:
                print(f"Model '{model_name}' does not have a predict method.")

    def roc_curves(self):
        """
        Plots the ROC Curves for all loaded models.
        """
        for model_name, model in self.models.items():
            if hasattr(model, 'predict'):
                predictions = model.predict(self.learner.X_test)
                if predictions.ndim > 1 and predictions.shape[1] > 1:  # Check if predictions are multi-class
                    predictions = np.argmax(predictions, axis=1)
                fpr, tpr, _ = roc_curve(self.learner.y_test, predictions)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve for {model_name}')
                plt.legend(loc="lower right")
                plt.show()
            else:
                print(f"Model '{model_name}' does not have a predict method.")

    def load_our_models(self):
        """
        Loads the models from the Model_Storage directory.
        """
        model_ANN = load_model(f"{self.model_folder_path}/ANN_model.h5")
        self.models['ANN'] = model_ANN

        model_DecisionTree = joblib.load(f"{self.model_folder_path}/Decision_Tree.pkl")
        self.models['DecisionTree'] = model_DecisionTree


