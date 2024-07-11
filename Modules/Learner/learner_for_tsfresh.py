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


class Learner:
    def __init__(self):
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
            data = pd.read_csv(
                r'C:\Users\CR\python_advanced_predictive_maintenance\CSV_Features\Merge_CSV.csv')
            print("Data loaded successfully")
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV: {e}")
            return

        # Drop 'file_name' column if it exists
        if 'file_name' in data.columns:
            data.drop(columns=['file_name'], inplace=True)

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
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train), y=self.y_train)
        class_weights = {class_index: weight for class_index, weight in enumerate(weights)}
        self.class_weights = class_weights

        # Print classes and their weights
        print("Classes and their weights:")
        for class_index, weight in class_weights.items():
            print(f"Class {class_index}: Weight = {weight}")

    def rebalancing_with_imblearn(self):
        # Apply undersampling to balance the classes in the training data

        undersample = RandomUnderSampler(sampling_strategy='majority')
        self.X_train_resampled, self.y_train_resampled = undersample.fit_resample(self.X_train, self.y_train)

        print(f"Num of class 0 in train set after undersampling: {np.count_nonzero(self.y_train_resampled == 0)}")
        print(f"Num of class 1 in train set after undersampling: {np.count_nonzero(self.y_train_resampled == 1)}")

    def standardize_features(self):
        # Select only numeric columns for standardization
        numeric_columns = self.X_train_resampled.select_dtypes(include=[np.number]).columns

        # Use these numeric columns for fitting and transforming scaler
        self.scaler = StandardScaler()
        self.X_train_resampled[numeric_columns] = self.scaler.fit_transform(self.X_train_resampled[numeric_columns])
        self.X_test[numeric_columns] = self.scaler.transform(self.X_test[numeric_columns])

        print("First row of scaled training set:", self.X_train_resampled.iloc[0])

    def build_model(self):
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
        model.fit(self.X_train_resampled, self.y_train_resampled, batch_size=32, epochs=50, callbacks=[early_stopping])

        # Save the trained model to a file
        model.save(f"{self.model_folder_path}/ANN_model.h5")
        self.predict_ANN = model.predict(self.X_test)

        # Convert probabilities to class labels (0 or 1) using a threshold of 0.5
        predict = np.where(self.predict_ANN > 0.5, 1, 0)

        # Compute accuracy score and print classification report
        self.ANN_accuracy = accuracy_score(self.y_test, predict)
        print(classification_report(self.y_test, predict))

    def run_DecisionTree(self):
        # Train a decision tree classifier
        model = DecisionTreeClassifier()
        model.fit(self.X_train_resampled, self.y_train_resampled)
        joblib.dump(model, f"{self.model_folder_path}/Decision_Tree.pkl")
        # Perform predictions on the test set
        self.predict_decision_tree = model.predict(self.X_test)
        self.D_tree_accuracy = accuracy_score(self.y_test, self.predict_decision_tree)
        print(f"Decision Tree Accuracy: {self.D_tree_accuracy}")

    def run_learner(self):
        self.rebalancing_with_class_weights()
        self.rebalancing_with_imblearn()
        self.standardize_features()
        self.run_DecisionTree()
        self.build_model()


# Example usage
if __name__ == "__main__":
    learner = Learner()
    learner.run_learner()
