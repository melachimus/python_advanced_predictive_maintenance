"""
Filename: Rockets.py
Author: Niklas Bukowski <bukowsni@hs-albsig.de>

Created at: 2024-07-13
Last changed: 2024-07-16
"""


import numpy as np
import pandas as pd
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from tsai.all import *
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import pickle

def save_plot(fig, title, subfolder="Rocket"):
    # Create directories if they don't exist
    base_dir = "Graphen"
    path = os.path.join(base_dir, subfolder)
    os.makedirs(path, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%y%d%m%H%M")
    filename = f"{title}_{timestamp}.png"
    
    # Save the plot
    filepath = os.path.join(path, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Plot saved to {filepath}")

def save_model(model, classifier_type, balanced_acc):
    # Create directories if they don't exist
    base_dir = "Models"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%y%d%m%H%M")
    filename = f"{classifier_type}_{timestamp}_BA_{balanced_acc:.2f}.pkl"
    
    # Save the model
    filepath = os.path.join(base_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filename):
    filepath = os.path.join("Models", filename)
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model

def fp_prep_data(file_path):
    print("Loading data from file...")
    df = pd.read_csv(file_path)
    print("Data loaded.")
    
    def y_func(o): 
        return o

    X, y = df2xy(df, sample_col='sample', feat_col='feature', target_col='target', data_cols=df.columns[2:-1], y_func=y_func)
    
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    
    splits = get_splits(y, valid_size=.2, stratify=True, random_state=42, shuffle=True, show_plot=False)
    X_train, y_train = X[splits[0]], y[splits[0]]
    X_test, y_test = X[splits[1]], y[splits[1]]

    print("Data preparation complete.")
    return X_train, y_train, X_test, y_test

def df_prep_data(df):
    print("Preparing data from DataFrame...")
    
    def y_func(o): 
        return o

    X, y = df2xy(df, sample_col='sample', feat_col='feature', target_col='target', data_cols=df.columns[2:-1], y_func=y_func)
    
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')
    
    splits = get_splits(y, valid_size=.2, stratify=True, random_state=42, shuffle=True)
    X_train, y_train = X[splits[0]], y[splits[0]]
    X_test, y_test = X[splits[1]], y[splits[1]]
    
    fig, ax = plt.subplots(figsize=(15, 3))
    plot_splits(splits, ax=ax, data=y)
    save_plot(fig, "Train_Test_Split_Distribution")

    print("Data preparation complete.")
    return X_train, y_train, X_test, y_test

def fit_predict(X_train, y_train, X_test, classifier_type='Rocket', load_model_name=None):
    if load_model_name:
        print(f"Loading model from {load_model_name}...")
        cls = load_model(load_model_name)
    else:
        print(f"Fitting {classifier_type} classifier...")
        if classifier_type.lower() == 'rocket':
            cls = RocketClassifier(random_state=42)
        elif classifier_type.lower() == 'minirocket':
            cls = MiniRocketClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
        cls.fit(X_train, y_train)
        print("Classifier fitted.")
    
    print("Making predictions...")
    y_pred = cls.predict(X_test)
    print("Predictions complete.")
    
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print("Saving model...")
    save_model(cls, classifier_type, balanced_acc)
    
    return y_pred

def calc_metrics(y_test, y_pred):
    print("Evaluating metrics...")
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f'Balanced Accuracy: {balanced_acc}')

    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'F1-Score: {f1}')

    precision = precision_score(y_test, y_pred, average='weighted')
    print(f'Precision: {precision}')

    recall = recall_score(y_test, y_pred, average='weighted')
    print(f'Recall: {recall}')

    conf_matrix = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    
    save_plot(fig, "Confusion_Matrix")
    
    class_report = classification_report(y_test, y_pred)
    print(f'Classification Report:\n{class_report}')
    print("Metrics evaluation complete.")

def show_results(X_test, y_test, y_pred):
    print("Visualizing classification results...")
    
    def plot_time_series(X, y_true, y_pred, indices, title):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        for i, idx in enumerate(indices):
            ax = axs[i // 2, i % 2]
            ax.plot(X[idx, 0], lw=2)
            ax.set_title(f'True: {y_true[idx]} \nPred: {y_pred[idx]}', color='green' if y_true[idx] == y_pred[idx] else 'red')
        fig.suptitle(title)
        plt.tight_layout()
        
        save_plot(fig, title)
    
    correct_indices = np.where(y_test == y_pred)[0]
    incorrect_indices = np.where(y_test != y_pred)[0]
    
    correct_indices = correct_indices[:4] if len(correct_indices) >= 4 else correct_indices
    incorrect_indices = incorrect_indices[:4] if len(incorrect_indices) >= 4 else incorrect_indices
    
    if len(correct_indices) > 0:
        plot_time_series(X_test, y_test, y_pred, correct_indices, "Correctly_Classified_Instances")
    
    if len(incorrect_indices) > 0:
        plot_time_series(X_test, y_test, y_pred, incorrect_indices, "Misclassified_Instances")
    
    print("Visualization complete.")

if __name__ == "__main__":
    print("Starting main script...")
    
    # # MIMII Pump_00:
    file_path = r"C:\Users\onlyf\OneDrive - Hochschule Albstadt-Sigmaringen\1FH\Study\#MSC WIW DPM\2nd Semerster\Piethong\Pyhton Advanced\Prüfungsleistung\predictive_maintenance\CSV\transformed_amplituden_dfs.csv"
    X_train, y_train, X_test, y_test = fp_prep_data(file_path)

    # SampleData.csv:
    # file_path = r"C:\Users\onlyf\OneDrive - Hochschule Albstadt-Sigmaringen\1FH\Study\#MSC WIW DPM\2nd Semerster\Piethong\Pyhton Advanced\Prüfungsleistung\predictive_maintenance\CSV\sample_data.csv"
    # X_train, y_train, X_test, y_test = fp_prep_data(file_path)

    # # MIMII UCR_data:
    # X_train, y_train, X_test, y_test = get_UCR_data('OliveOil', Xdtype='float64')

    # To load an existing model, specify the filename here:
    load_model_name = None  # Set to the filename if you want to load a model, e.g., "RocketClassifier_202207011230.pkl"

    # Choose between 'Rocket' and 'MiniRocket', case-insensitive
    classifier_type = 'Rocket'

    y_pred = fit_predict(X_train, y_train, X_test, classifier_type, load_model_name)
    calc_metrics(y_test, y_pred)
    show_results(X_test, y_test, y_pred)
    print("Main script execution finished.")


