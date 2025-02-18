'''Define machine learning pipeline for training, evaluating, and using deep learning models with TensorFlow/Keras.'''

import os

from .feature_engineer import np, pd

from keras.callbacks import EarlyStopping
import keras_tuner as kt
from keras.optimizers import Adam

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Activation


from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

from .constants import checkpoint_path, log_dir
from typing import Dict


checkpoint_dir = os.path.dirname(checkpoint_path)



class Neural_Network:
    '''Define machine learning pipeline for training, evaluating, and using deep learning models with TensorFlow/Keras.

    Attributes:
    - model (Sequential): Stores model instance.
    - is_trained (bool): Tracks whether the model has been trained.
    - input_shape (tuple): Shape of the input data.
    - best_hps (Dict): Stores the best hyperparameters.
    '''

    def __init__(self):
        self.model = None
        self.is_trained = False
        self.input_shape = None
        self.tune_parameters = True
        self.best_hps = None

    def _build_model(self):
        '''Build a Dense Neural Network model for non-sequential data.

        Arguments:
        - input_shape (tuple): Shape of the input data.

        Returns:
        - Sequential: Compiled Dense Neural Network model.
        '''
        model = Sequential()
        model.add(Dense(self.best_hps.get('units_input'), activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(self.best_hps.get('dropout_input')))
        model.add(Dense(self.best_hps.get('units_hidden_1'), activation='relu'))
        model.add(Dropout(self.best_hps.get('dropout_hidden_1')))
        model.add(Dense(self.best_hps.get('units_hidden_2'), activation='relu'))
        model.add(Dense(3, activation='softmax'))
        return model

    def _create_model(self) -> None:
        '''Create and compile a Dense Neural Network model for non-sequential data.

        Arguments:
        - X_train (np.ndarray): The training input data to determine the input shape.
        - learning_rate (float): Learning rate for the Adam optimizer. Defaults to 0.001.
        - beta_1 (float): Exponential decay rate for the first moment estimate. Defaults to 0.99.
        - beta_2 (float): Exponential decay rate for the second moment estimate. Defaults to 0.9999.
        - epsilon (float): Small constant for numerical stability. Defaults to 1e-07.

        Attributes Updated:
        - self.model: Stores the compiled Dense Neural Network model.
        - self.is_trained: Set to True after model creation.

        Returns:
        - None
        '''
        model = self._build_model()
        model.compile(        
            optimizer=Adam(
            learning_rate=self.best_hps.get('learning_rate'),
            beta_1=self.best_hps.get('beta_1'),
            beta_2=self.best_hps.get('beta_2'),
            epsilon=self.best_hps.get('epsilon')
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
        self.model = model
        self.is_trained = True
    
    def _model_choice(self, hp) -> Sequential:
        '''Define the hyperparameter search space for the model.

        Arguments:
        - hp: Hyperparameters to search.

        Returns:
        - Sequential: Compiled Dense Neural Network model.
        '''
        model = Sequential()
        model.add(Dense(hp.Int('units_input', min_value=64, max_value=256, step=32),
                        activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout_input', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(hp.Int('units_hidden_1', min_value=32, max_value=128, step=16), activation='relu'))
        model.add(Dropout(hp.Float('dropout_hidden_1', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(hp.Int('units_hidden_2', min_value=16, max_value=64, step=8), activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(
            optimizer=Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]),
            beta_1=hp.Choice('beta_1', [0.8, 0.9, 0.99]),
            beta_2=hp.Choice('beta_2', [0.9, 0.999, 0.9999]),
            epsilon=hp.Choice('epsilon', [1e-5, 1e-6, 1e-7, 1e-8])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )
        return model

    def _get_hyperparameters(self, X_train, y_train, X_val, y_val) -> None:
        '''Get the best hyperparameters for the model.
        
        Arguments:
        - hp: Hyperparameters to search.
        - X_train (np.ndarray): Training input data.
        - y_train (pd.Series): Training target labels.
        - X_val (np.ndarray): Validation input data.
        - y_val (pd.Series): Validation target labels.
        
        Returns:
        - Dict: Best hyperparameters for the model.
        '''
        tuner = kt.Hyperband(
            self._model_choice,
            objective='val_accuracy',
            max_epochs=20,
            directory='./sport_bets/models/hyperparameters',
            project_name='tunner_search'
        )
        tuner.search(
            X_train, y_train, epochs=50,
            validation_data=(X_val, y_val),
            )
        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


    def _train_model(self, 
                     X_train: np.ndarray, X_test: np.ndarray,
                     y_train: pd.Series, y_test: pd.Series, 
                     callback = None) -> None:
        '''Train model using K-fold cross-validation.

        Arguments:
        - X_train (np.ndarray): Training input data.
        - X_test (np.ndarray): Test input data.
        - y_train (pd.Series): Training target labels.
        - y_test (pd.Series): Test target labels.
        - n_splits (int): Number of K-fold splits for cross-validation. Defaults to 3.

        Returns:
        - None

        Additional Notes:
        - Updates the model after training each fold.
        - Calculates log loss and accuracy for each fold.
        '''
        early_stopping = EarlyStopping(monitor='val_loss', 
                                patience=10,    
                                min_delta=0.001,     
                                mode='min',
                                restore_best_weights=True)
        if callback is None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            verbose=1)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                                histogram_freq=1)
            callbacks = [early_stopping, cp_callback, tensorboard_callback]
        else: 
            callbacks = [early_stopping, callback]  
        X_train_fold, y_train_fold = X_train, y_train
        X_val_fold, y_val_fold = X_test, y_test
        self.input_shape = X_train_fold.shape[1:]
        self._get_hyperparameters(X_train_fold, y_train_fold, 
                                  X_val_fold, y_val_fold)
        self._create_model()
        self.model.fit(X_train_fold, y_train_fold, 
                    epochs=50, batch_size=8, 
                    verbose=1, 
                    callbacks = callbacks,
                    validation_data=(X_val_fold, y_val_fold))
        y_pred = self.model.predict(X_test)
        log_loss_fold = log_loss(y_test, y_pred)
        accuracy_fold = accuracy_score(y_test, np.argmax(y_pred, axis=1))

        print(f"Log Loss: {log_loss_fold}")
        print(f'Accuracy: {accuracy_fold}')

    def _predict(self, X: np.ndarray) -> np.ndarray:
        '''Make predictions on the input data.

        Arguments:
        - X (np.ndarray): Input data for predictions.

        Returns:
        - y_pred (np.ndarray): Predicted probabilities or outputs from the model.
        '''
        y_pred = self.model.predict(X)
        return y_pred
    
    def _load_model(self, X_train_reshaped: np.ndarray) -> None:
        '''Load model weights from a saved checkpoint.

        Arguments:
        - X_train_reshaped (np.ndarray): Input data to initialize the model shape.

        Attributes Updated:
        - model: Recreated model loaded with saved weights.
        - is_trained: Set to True after loading the weights.

        Returns:
        - None
        '''
        self.input_shape = X_train_reshaped.shape()
        self.model = self._create_model()
        self.model.load_weights(checkpoint_path)
        self.is_trained = True
