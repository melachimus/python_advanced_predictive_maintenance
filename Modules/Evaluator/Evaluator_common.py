import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
import os
from Modules.Learner.learner import Learner  # Ensure correct import path

# ID:02 (ChatGPT)
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
        results = pd.DataFrame(sorted_models, columns=["Model", "Accuracy"])
        with open(os.path.join(self.model_folder_path, "best_accuracy_manual_extraction.txt"), "w") as file:
            file.write(results.to_string(index=False))
        
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
        for file_name in os.listdir(self.model_folder_path):
            model_path = os.path.join(self.model_folder_path, file_name)
            if file_name.endswith('.h5'):
                model = load_model(model_path)
                self.models[file_name.split('.')[0]] = model
            elif file_name.endswith('.pkl'):
                model = joblib.load(model_path)
                self.models[file_name.split('.')[0]] = model
            elif file_name.endswith(".pth"):
                print("wird über evaluator_pytorch.py erledigt")
            elif file_name.endswith(".txt"):
                pass
            elif file_name.endswith(".png"):
                pass
            else:
                print(f"Unsupported file type: {file_name}")



