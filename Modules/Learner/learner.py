import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
from keras import callbacks
import joblib
import matplotlib.pyplot as plt


class Learner:
    """
    A class for loading data, preprocessing, rebalancing, and training classification models.

    Attributes
    ----------
    X_train : pd.DataFrame or None
        Training features.
    X_test : pd.DataFrame or None
        Test features.
    y_train : np.array or None
        Training target.
    y_test : np.array or None
        Test target.
    class_weights : dict or None
        Class weights for rebalancing.
    X_train_resampled : pd.DataFrame or None
        Resampled training features after balancing.
    y_train_resampled : np.array or None
        Resampled training target after balancing.
    scaler : StandardScaler or None
        Scaler object for standardizing features.
    predict_ANN : np.array or None
        Predictions from the trained neural network model.
    ANN_accuracy : float or None
        Accuracy score of the neural network model.
    predict_decision_tree : np.array or None
        Predictions from the trained decision tree model.
    D_tree_accuracy : float or None
        Accuracy score of the decision tree model.

    Methods
    -------
    __init__(self, csv_path):
        Initializes attributes and loads data from the given CSV file.
    rebalancing_with_class_weights(self):
        Computes class weights and assigns them to self.class_weights.
    rebalancing_with_imblearn(self):
        Performs undersampling on the majority class in the training data.
    standardize_features(self):
        Standardizes numeric features in the resampled training set and test set.
    build_model(self):
        Builds and trains a neural network model using Keras.
    run_DecisionTree(self):
        Trains a decision tree classifier and evaluates its performance on the test set.
    run_learner(self):
        Executes the learning pipeline by invoking rebalancing, feature standardization, and model training.
    """

    def __init__(self, csv_path):
        """
        Initializes the Learner with attributes set to None and loads data from the specified CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing the data.
        """
        # Initialize attributes as None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_weights = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.scaler = None
        self.predict_ANN = None
        self.ANN_accuracy = None
        self.predict_decision_tree = None
        self.D_tree_accuracy = None

        # Load data with exception handling
        try:
            data = pd.read_csv(csv_path)
            print("Data loaded successfully")
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV: {e}")
            return

        # Determine current script directory
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        # Navigate two levels up from current script to 'Gruppe3' folder
        base_folder_path = os.path.join(current_script_path, '..', '..')
        self.model_folder_path = os.path.join(base_folder_path, "Model_Storage")

        # Prepare features and target
        features = data.drop(['Label'], axis=1)
        target = data['Label']

        # Encode categorical target labels to numerical values
        label_encoder = LabelEncoder()
        target_encoded = label_encoder.fit_transform(target)

        X = features
        y = target_encoded

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                                                stratify=y)

        # Print the shape of the features
        print("Shape of X:", X.shape)

    def rebalancing_with_class_weights(self):
        """
        Computes class weights for imbalanced classes in self.y_train and assigns them to self.class_weights.

        The class weights are computed using sklearn's compute_class_weight function based on the class distribution
        in self.y_train.
        """
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weights = {class_index: weight for class_index, weight in enumerate(weights)}
        self.class_weights = class_weights

        # Print classes and their weights
        print("Classes and their weights:")
        for class_index, weight in class_weights.items():
            print(f"Class {class_index}: Weight = {weight}")

    def rebalancing_with_imblearn(self):
        """
        Performs undersampling on the majority class in self.X_train and self.y_train using RandomUnderSampler from imblearn.

        The majority class is undersampled to balance the class distribution in the training data.
        """
        # Apply undersampling to balance the classes in the training data
        undersample = RandomUnderSampler(sampling_strategy='majority')
        self.X_train_resampled, self.y_train_resampled = undersample.fit_resample(self.X_train, self.y_train)

        print(f"Num of class 0 in train set after undersampling: {np.count_nonzero(self.y_train_resampled == 0)}")
        print(f"Num of class 1 in train set after undersampling: {np.count_nonzero(self.y_train_resampled == 1)}")

    def standardize_features(self):
        """
        Standardizes numeric features in self.X_train_resampled and self.X_test using StandardScaler.

        Only numeric columns are selected for standardization, and the scaler is fitted on self.X_train_resampled.
        """
        # Select only numeric columns for standardization
        numeric_columns = self.X_train_resampled.select_dtypes(include=[np.number]).columns

        # Use these numeric columns for fitting and transforming scaler
        self.scaler = StandardScaler()
        self.X_train_resampled[numeric_columns] = self.scaler.fit_transform(self.X_train_resampled[numeric_columns])
        self.X_test[numeric_columns] = self.scaler.transform(self.X_test[numeric_columns])

        print("First row of scaled training set:", self.X_train_resampled.iloc[0])

    def build_model(self):
        """
        Builds and trains a neural network model using Keras Sequential API.

        The model architecture consists of several dense layers with dropout for regularization. The model is compiled
        with Adam optimizer and binary crossentropy loss. Training stops early if the loss does not improve after a
        certain number of epochs.
        """
        # Define early stopping to prevent overfitting
        early_stopping = callbacks.EarlyStopping(
            monitor='loss',
            min_delta=0.001,
            patience=20,
            restore_best_weights=True
        )

        # Build a neural network model using Keras Sequential API
        model = Sequential()
        model.add(Dense(units=600, kernel_initializer='uniform', activation='relu',
                        input_dim=self.X_train_resampled.shape[1]))
        model.add(Dense(units=500, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=400, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(units=200, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

        opt = Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history=model.fit(self.X_train_resampled, self.y_train_resampled, batch_size=32, epochs=50, callbacks=[early_stopping], validation_split =0.2)
        history_df = pd.DataFrame(history.history)

        plt.plot(history_df.loc[:, ['loss']], "#BDE2E2", label='Training loss')
        plt.plot(history_df.loc[:, ['val_loss']],"#C2C4E2", label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc="best")

        plt.show()

        history_df = pd.DataFrame(history.history)

        plt.plot(history_df.loc[:, ['accuracy']], "#BDE2E2", label='Training accuracy')
        plt.plot(history_df.loc[:, ['val_accuracy']], "#C2C4E2", label='Validation accuracy')

        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Save the trained model to a file
        model.save(f"{self.model_folder_path}/ANN_model.h5")
        self.predict_ANN = model.predict(self.X_test)

        # Convert probabilities to class labels (0 or 1) using a threshold of 0.5
        predict = np.where(self.predict_ANN > 0.5, 1, 0)

        # Compute accuracy score and print classification report
        self.ANN_accuracy = accuracy_score(self.y_test, predict)
        print(classification_report(self.y_test, predict))

    def run_DecisionTree(self):
        """
        Trains a decision tree classifier on self.X_train_resampled and self.y_train_resampled.

        The trained model is saved to a file using joblib. Performance metrics such as accuracy are computed and printed.
        """
        # Train a decision tree classifier
        model = DecisionTreeClassifier()
        model.fit(self.X_train_resampled, self.y_train_resampled)
        joblib.dump(model, f"{self.model_folder_path}/Decision_Tree.pkl")

        # Perform predictions on the test set
        self.predict_decision_tree = model.predict(self.X_test)
        self.D_tree_accuracy = accuracy_score(self.y_test, self.predict_decision_tree)
        print(f"Decision Tree Accuracy: {self.D_tree_accuracy}")
        print(classification_report(self.y_test,self.predict_decision_tree))

    def run_learner(self):
        """
        Executes the learning pipeline by invoking rebalancing, feature standardization, and model training.

        This method sequentially calls other methods to rebalance classes, standardize features, and train both a
        neural network model and a decision tree classifier.
        """
        self.rebalancing_with_class_weights()
        self.rebalancing_with_imblearn()
        self.standardize_features()
        self.run_DecisionTree()
        self.build_model()



