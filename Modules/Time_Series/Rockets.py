"""
Filename: Rockets.py
Author: Niklas Bukowski <bukowsni@hs-albsig.de>

Version: Relative Paths, Classes, Run Directory

Created at: 2024-07-13
Last changed: 2024-07-16
"""

import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tsai.all import *
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

class RunManager:
    def __init__(self, base_dir, dataset_name, classifier_type):
        self.base_dir = os.path.join(base_dir, "rocket_runs")
        os.makedirs(self.base_dir, exist_ok=True)
        self.run_dir = os.path.join(self.base_dir, self.create_run_folder_name(dataset_name, classifier_type))
        os.makedirs(self.run_dir, exist_ok=True)

    def create_run_folder_name(self, dataset_name, classifier_type):
        timestamp = datetime.now().strftime("%y%m%d%H%m")
        return f"{timestamp}_{dataset_name}_{classifier_type}"

    def get_run_path(self):
        return self.run_dir
    
    def get_run_folder_name(self):
        return os.path.basename(self.run_dir)
    
class DataHandler:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def load_data(self, file_path):
        full_path = os.path.join(self.base_dir, file_path)
        print("Loading data from file...")
        df = pd.read_csv(full_path)
        print("Data loaded.")
        return df

    def prep_data(self, df, sample_col, feat_col, target_col, data_cols, y_func=lambda o: o):
        X, y = df2xy(df, sample_col=sample_col, feat_col=feat_col, target_col=target_col, data_cols=data_cols, y_func=y_func)
        print(f'X shape: {X.shape}, y shape: {y.shape}')
        splits = get_splits(y, valid_size=.2, stratify=True, random_state=42, shuffle=True, show_plot=False)
        X_train, y_train = X[splits[0]], y[splits[0]]
        X_test, y_test = X[splits[1]], y[splits[1]]
        print("Data preparation complete.")
        return X_train, y_train, X_test, y_test

class ModelHandler:
    def __init__(self, classifier_type='Rocket', random_state=42, run_manager = None):
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.model = None
        self.run_manager = run_manager


    def fit_predict(self, X_train, y_train, X_test, model_file_name=None):  # Updated function signature
        if model_file_name:  # Check if loading an existing model
            self.model = self.load_model(model_file_name, self.run_manager.base_dir)  # Load the existing model
        else:
            if self.classifier_type.lower() == 'rocket':
                self.model = RocketClassifier(random_state=self.random_state)
            elif self.classifier_type.lower() == 'minirocket':
                self.model = MiniRocketClassifier(random_state=self.random_state)
            else:
                raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
            self.model.fit(X_train, y_train)
            print(f"{self.classifier_type} classifier fitted.")
        
        y_pred = self.model.predict(X_test)
        print("Prediction made.")
        return y_pred

    def save_model(self, balanced_acc):
        timestamp = datetime.now().strftime("%y%m%d%H%m")  # Get current datetime as a string
        filename = f"{self.classifier_type}_BA_{balanced_acc:.2f}_{timestamp}.pkl"  # Include timestamp in filename
        filepath = os.path.join(self.run_manager.get_run_path(), filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to Run Directory.")

    def load_model(self, model_file_name, base_dir):
        for root, dirs, files in os.walk(base_dir):
            if model_file_name in files:
                filepath = os.path.join(root, model_file_name)
                with open(filepath, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Model loaded from {filepath}")
                return self.model
        raise FileNotFoundError(f"File {model_file_name} not found in {base_dir}")

class Evaluation:
    def __init__(self, y_test, y_pred, X_test, run_manager = None):
        self.y_test = y_test
        self.y_pred = y_pred
        self.X_test = X_test
        self.balanced_acc = None
        self.metrics_report = ""
        self.run_manager = run_manager

        self.calculate_metrics()  # Calculate all metrics and classification report
        self.save_metrics()  # Save metrics and classification report

        # Save outputs and generate reports
        self.generate_confusion_matrix()
        self.show_results("Correctly_Classified_Instances")
        self.show_results("Misclassified_Instances")
        print("Evaluation saved in Run Directory.")

    def calculate_metrics(self):
        # Calculate metrics
        self.balanced_acc = balanced_accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred, average='weighted')
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        recall = recall_score(self.y_test, self.y_pred, average='weighted')

        # Get classification report
        report = classification_report(self.y_test, self.y_pred)

        # Build the output string
        self.metrics_report = (
            f'Balanced Accuracy: {self.balanced_acc}\n'
            f'F1-Score: {f1}\n'
            f'Precision: {precision}\n'
            f'Recall: {recall}\n\n'
            f'Classification Report:\n{report}'
        )

    def save_metrics(self):
        run_folder_name = self.run_manager.get_run_folder_name()
        title = f"{run_folder_name}\n\n"
        full_report = title + self.metrics_report

        filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(self.run_manager.get_run_path(), filename)
        with open(filepath, 'w') as f:
            f.write(full_report)

    def generate_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test), ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        self.save_plot(fig, "Confusion_Matrix")

    def show_results(self, title):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        correct_indices = np.where(self.y_test == self.y_pred)[0]
        incorrect_indices = np.where(self.y_test != self.y_pred)[0]
        indices = correct_indices if 'Correct' in title else incorrect_indices
        for i, idx in enumerate(indices[:4]):
            ax = axs[i // 2, i % 2]
            ax.plot(self.X_test[idx, 0], lw=2)
            ax.set_title(f'True: {self.y_test[idx]} \nPred: {self.y_pred[idx]}', color='green' if self.y_test[idx] == self.y_pred[idx] else 'red')
        fig.suptitle(title)
        plt.tight_layout()
        self.save_plot(fig, title)

    def save_plot(self, fig, title, subfolder=""):
        path = os.path.join(self.run_manager.get_run_path(), subfolder)
        os.makedirs(path, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d%H%m")
        filename = f"{title}_{timestamp}.png"
        filepath = os.path.join(path, filename)
        fig.savefig(filepath)
        plt.close(fig)

if __name__ == "__main__":
    
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_handler = DataHandler(current_dir)

    still_testing = True
    if still_testing == True:
        #Select Dataset
        dataset = "OliveOil"

        if dataset == "OliveOil":
            # UCR_data OliveOil:
            X_train, y_train, X_test, y_test = get_UCR_data('OliveOil', Xdtype='float64')

        elif dataset == "MIMII":
            # Provide relative path for the data file
            file_path = "CSV\\sample_data.csv"       #"CSV\\transformed_amplituden_dfs.csv"
            df = data_handler.load_data(file_path)
            X_train, y_train, X_test, y_test = data_handler.prep_data(df, sample_col='sample', feat_col='feature', target_col='target', data_cols=df.columns[2:-1])
        
        classifier_type = 'Rocket'

    elif still_testing == False:
        config_file = os.path.join(os.path.dirname(current_dir), 'config.json')
        config = read_config(config_file)
        time_series_file = os.path.join(current_dir, config['amplitude_file'].replace('/', os.path.sep))
        time_series = read_time_series(time_series_file)
        transformed_dataset = transform_dataset(time_series, sample='file_name', target='Label', value='amplitude')
        X_train, y_train, X_test, y_test = data_handler.prep_data(transformed_dataset, sample_col='sample', feat_col='feature', target_col='target', data_cols=df.columns[2:-1])

    run_manager = RunManager(current_dir, dataset, classifier_type)
    model_handler = ModelHandler(classifier_type, 42, run_manager)

    model_file_name = None      #Choose None for training a new Model or load one like this "MiniRocket_BA_0.55.pkl"
    #run rocket
    y_pred = model_handler.fit_predict(X_train, y_train, X_test, model_file_name )
    
    # Calculate and save metrics
    eval = Evaluation(y_test, y_pred, X_test, run_manager)

    # Save Model
    model_handler.save_model(eval.balanced_acc)

    print("Main script execution finished.")