"""
Filename: inception_time.py
Author: Luca-David Stegmaier <stegmalu@hs-albsig.de>

Created at: 2024-07-17
Last changed: 2024-07-17
"""
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
from tsai.all import *
from sklearn import metrics
from preprocessor_tseries import transform_dataset, read_config, read_time_series
from Rockets import DataHandler, RunManager

class InceptionTimeModel:
    def __init__(self, base_dir: str, config: dict):
        """
        Initialisiert das InceptionTimeModel mit den angegebenen Basisverzeichnis und der Konfiguration.

        Args:
            base_dir (str): Basisverzeichnis für Modell und Läufe.
            config (dict): Konfigurationsparameter.
        """
        self.base_dir = base_dir
        self.config = config
        self.model_storage_path = os.path.join(base_dir, "Model_Storage")
        os.makedirs(self.model_storage_path, exist_ok=True)
        self.model_path = os.path.join(self.model_storage_path, "InceptionTime.pkl")
        self.run_dir = self.create_run_directory()
        os.makedirs(self.run_dir, exist_ok=True)
        self.learner = None

    def create_run_directory(self) -> str:
        """
        Erstellt ein Verzeichnis für einen neuen Lauf basierend auf dem aktuellen Zeitstempel.

        Returns:
            str: Pfad zum erstellten Verzeichnis.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        run_dir = os.path.join(current_dir, "runs_inception_time", f"InceptionTime_{timestamp}")
        return run_dir

    def load_or_train_model(self, dataloaders: TSDataLoaders, params: dict) -> None:
        """
        Lädt ein vorhandenes Modell oder trainiert ein neues, falls kein Modell vorhanden ist.

        Args:
            dataloaders (TSDataLoaders): Datenlader für Trainings- und Validierungsdaten.
            params (dict): Trainingsparameter wie Epoche, Batch-Größe und Lernrate.
        """
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            try:
                with open(self.model_path, 'rb') as f:
                    self.learner = pickle.load(f)
            except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
                print(f"Error loading model: {e}")
                self.train_and_save_model(dataloaders, params)
        else:
            self.train_and_save_model(dataloaders, params)
    
    def train_and_save_model(self, dataloaders: TSDataLoaders, params: dict) -> None:
        """
        Trainiert das Modell und speichert es anschließend.

        Args:
            dataloaders (TSDataLoaders): Datenlader für Trainings- und Validierungsdaten.
            params (dict): Trainingsparameter wie Epoche, Batch-Größe und Lernrate.
        """
        print(f"Training model and saving to {self.model_path}")
        model = InceptionTime(dataloaders.vars, dataloaders.c)
        self.learner = Learner(dataloaders, model, metrics=accuracy)
        self.learner.fit_one_cycle(params['epochs'], lr_max=params['lr'])
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.learner, f)

    def save_plot(self, fig: plt.Figure, name: str) -> None:
        """
        Speichert die gegebene Matplotlib-Abbildung unter dem angegebenen Namen.

        Args:
            fig (plt.Figure): Matplotlib-Abbildung.
            name (str): Name der Abbildung.
        """
        if fig is not None:
            filename = os.path.join(self.run_dir, f"InceptionTime_{name}.png")
            fig.savefig(filename)
            plt.close(fig)
            print(f"Saved {filename}")
        else:
            print(f"No figure to save for {name}")

    def run(self, csv_path: str) -> None:
        """
        Führt den kompletten Trainings- und Evaluationsprozess aus.

        Args:
            csv_path (str): Pfad zur CSV-Datei mit den Amplitudendaten.
        """
        config_file = os.path.join(self.base_dir, 'config.json')
        config = read_config(config_file)

        time_series_file = os.path.join(self.base_dir, config['amplitude_file'].replace('/', os.path.sep))
        time_series = read_time_series(time_series_file)

        transformed_dataset = transform_dataset(time_series, sample='file_name', target='Label', value='amplitude')

        handler = DataHandler(self.base_dir)
        X_train, y_train, X_test, y_test = handler.prep_data(transformed_dataset, sample_col='sample',
                                                             feat_col='feature',
                                                             target_col='target',
                                                             data_cols=transformed_dataset.columns[2:-1])
           

        print("Train features shape:", X_train.shape)
        print("Train targets shape:", y_train.shape)
        print("Test features shape:", X_test.shape)
        print("Test targets shape:", y_test.shape)

        # Check for NaN values and handle them
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0)

        X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])

        transforms = [None, [Categorize()]]
        datasets = TSDatasets(X, y, tfms=transforms, splits=splits, inplace=True)

        dataloaders = TSDataLoaders.from_dsets(datasets.train, datasets.valid,
                                               bs=[self.config['param_grid_inceptiontime']['batch_size'][0], self.config['param_grid_inceptiontime']['batch_size'][1]],
                                               batch_tfms=[TSStandardize()],
                                               num_workers=0)

        params = {
            'epochs': self.config['param_grid_inceptiontime']['epochs'][0], 
            'batch_size': self.config['param_grid_inceptiontime']['batch_size'][0],
            'lr': self.config['param_grid_inceptiontime']['lr'][0]
        }

        self.load_or_train_model(dataloaders, params)

        self.learner.show_results()
        fig = plt.gcf()
        self.save_plot(fig, "results")

        interp = ClassificationInterpretation.from_learner(self.learner)
        
        interp.plot_confusion_matrix()
        fig = plt.gcf()
        self.save_plot(fig, "confusion_matrix")

        most_confused = interp.most_confused(min_val=3)

        valid_dl = dataloaders.valid
        test_ds = valid_dl.dataset.add_test(X, y)
        test_dl = valid_dl.new(test_ds)
        next(iter(test_dl))

        test_probas, test_targets, test_preds = self.learner.get_preds(dl=test_dl,
                                                                      with_decoded=True,
                                                                      save_preds=None,
                                                                      save_targs=None)

        accuracy = metrics.accuracy_score(test_targets, test_preds)
        precision = metrics.precision_score(test_targets, test_preds, average='weighted')
        balanced_accuracy = metrics.balanced_accuracy_score(test_targets, test_preds)
        f1_score = metrics.f1_score(test_targets, test_preds, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"F1 Score: {f1_score:.4f}")

        self.save_model_info(accuracy, precision, balanced_accuracy, f1_score, most_confused)

    def save_model_info(self, accuracy: float, precision: float, balanced_accuracy: float, f1_score: float, most_confused: list) -> None:
        """
        Speichert Modellinformationen und Metriken in eine Textdatei.

        Args:
            accuracy (float): Genauigkeit des Modells.
            precision (float): Präzision des Modells.
            balanced_accuracy (float): Balancierte Genauigkeit des Modells.
            f1_score (float): F1-Score des Modells.
            most_confused (list): Liste der am meisten verwirrten Klassen.
        """
        info_path = os.path.join(self.model_storage_path, "InceptionTime_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
            f.write(f"F1 Score: {f1_score:.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(f"{most_confused}\n")
        print(f"Model info saved to {info_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, '..', '..')
    config_file = os.path.join(base_dir, 'config.json')
    config_file = os.path.abspath(config_file)
    
    config = read_config(config_file)
    csv_path = config['amplitude_file'].replace('/', os.path.sep)
    model = InceptionTimeModel(base_dir, config)
    model.run(csv_path)
