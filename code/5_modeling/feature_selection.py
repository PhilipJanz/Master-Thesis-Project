
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector
from scipy.stats import kendalltau
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


class AutoencoderFeatureSelector(tf.keras.Model):
    def __init__(self, input_dim, encoder_layers, decoder_layers=None):
        super(AutoencoderFeatureSelector, self).__init__()

        if decoder_layers is None:
            decoder_layers = [input_dim]  # Default to a single decoding layer

        # Define the encoder
        encoder_input = Input(shape=(input_dim,))
        encoded_output = encoder_input
        for units in encoder_layers:
            encoded_output = Dense(units, activation='relu')(encoded_output)
        self.encoded = encoded_output

        # Define the decoder
        decoder_input = self.encoded
        for units in decoder_layers[:-1]:  # Exclude the last layer
            decoder_input = Dense(units, activation='relu')(decoder_input)
        decoder_output = Dense(decoder_layers[-1], activation='linear')(decoder_input)

        # Create the encoder model
        self.encoder_model = Model(inputs=encoder_input, outputs=self.encoded)

        # Create the autoencoder model
        self.autoencoder_model = Model(inputs=encoder_input, outputs=decoder_output)

        # Compile the autoencoder model
        self.autoencoder_model.compile(optimizer='adam', loss='mean_absolute_error')

        # Initialize history dictionary
        self.history = {'loss': [], 'val_loss': []}

    def run(self, X):
        return self.autoencoder_model.predict(X)

    def encode(self, X):
        # Use the encoder model to transform the data
        return self.encoder_model.predict(X)

    def fit(self, X_train, X_target, X_test=None, epochs=10, batch_size=10, shuffle=True):
        # Train the autoencoder
        print("Training the autoencoder...")
        history = self.autoencoder_model.fit(
            X_train, X_target,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=1,
            validation_data=(X_test, X_test) if X_test is not None else None
        )

        # Update the history
        self.history['loss'].extend(history.history['loss'])
        if 'val_loss' in history.history:
            self.history['val_loss'].extend(history.history['val_loss'])
        else:
            self.history['val_loss'].extend([np.nan] * epochs)

    def plot_history(self, start_epoch=0):
        """
        Plot training and validation loss values starting from the specified epoch.

        Parameters:
        - start_epoch: The epoch from which to start the plot. Must be >= 0.
        """
        # Ensure start_epoch is valid
        if start_epoch < 0:
            raise ValueError("start_epoch must be >= 0")
        if start_epoch >= len(self.history['loss']):
            raise ValueError("start_epoch exceeds the number of epochs")

        # Calculate the range of epochs to plot
        epochs_range = range(start_epoch, len(self.history['loss']))

        # Get the subset of history starting from start_epoch
        loss_values = self.history['loss'][start_epoch:]
        val_loss_values = self.history['val_loss'][start_epoch:] if 'val_loss' in self.history else [np.nan] * len(loss_values)

        # Plot training & validation loss values
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, loss_values, label='Training Loss')
        if val_loss_values:
            plt.plot(epochs_range, val_loss_values, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # Set the y-axis to a logarithmic scale
        plt.legend()
        plt.show()


class FeatureLearner():
    def __init__(self, input_dim, layers):
        # Define the encoder
        input = Input(shape=(input_dim,))
        encoded_output = input
        for units in layers:
            encoded_output = Dense(units, activation='sigmoid')(encoded_output)
        predictor_output = Dense(1, activation='linear')(encoded_output)

        # Create the autoencoder model
        self.model = Model(inputs=input, outputs=predictor_output)

        # Compile the autoencoder model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Initialize history dictionary
        self.history = {'loss': [], 'val_loss': []}

    def predict(self, X):
        return self.model.predict(X)

    def get_features(self, X):
        last_hidden_layer_output = self.model.layers[-2].output  # -2 because -1 is the output layer
        self.feature_model = Model(inputs=self.model.input, outputs=last_hidden_layer_output)
        # Use the encoder model to transform the data
        return self.feature_model.predict(X)

    def fit(self, train_data, validation_data=None, epochs=10, batch_size=10, shuffle=True):
        # Train the autoencoder
        print("Training the autoencoder...")
        history = self.model.fit(
            train_data[0], train_data[1],
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=1,
            validation_data=validation_data if validation_data is not None else None
        )

        # Update the history
        self.history['loss'].extend(history.history['loss'])
        if 'val_loss' in history.history:
            self.history['val_loss'].extend(history.history['val_loss'])
        else:
            self.history['val_loss'].extend([np.nan] * epochs)

    def plot_history(self, start_epoch=0):
        """
        Plot training and validation loss values starting from the specified epoch.

        Parameters:
        - start_epoch: The epoch from which to start the plot. Must be >= 0.
        """
        # Ensure start_epoch is valid
        if start_epoch < 0:
            raise ValueError("start_epoch must be >= 0")
        if start_epoch >= len(self.history['loss']):
            raise ValueError("start_epoch exceeds the number of epochs")

        # Calculate the range of epochs to plot
        epochs_range = range(start_epoch, len(self.history['loss']))

        # Get the subset of history starting from start_epoch
        loss_values = self.history['loss'][start_epoch:]
        val_loss_values = self.history['val_loss'][start_epoch:] if 'val_loss' in self.history else [np.nan] * len(loss_values)

        # Plot training & validation loss values
        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, loss_values, label='Training Loss')
        if val_loss_values:
            plt.plot(epochs_range, val_loss_values, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')  # Set the y-axis to a logarithmic scale
        plt.legend()
        plt.show()


def backwards_feature_selection(X, y, feature_names):
    assert X.shape[1] == len(feature_names)

    # backward selection feature selection
    backward_fs = SequentialFeatureSelector(XGBRegressor(), forward=False, k_features="best")

    X_select = backward_fs.fit_transform(X, y)
    selected_feature_names = [feature_names[i] for i in backward_fs.k_feature_idx_]

    return X_select, selected_feature_names, backward_fs


def calculate_vif(X, feature_names):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature.

    Parameters:
    X : ndarray or DataFrame
        The matrix of features.
    feature_names : list
        The list of feature names.

    Returns:
    DataFrame
        DataFrame containing the VIF values for each feature.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)

    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['vif'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data


def feature_selection_corr_test(X_train, X_test, y_train, feature_names, indicator_names, alpha=0.1):
    selected_feature_ix = np.repeat(True, len(feature_names))
    indicator_feature_ls = []
    for i, feature in enumerate(feature_names):
        if feature not in indicator_names:
            continue
        corr, p_value = kendalltau(X_train[:, i], y_train)
        if p_value > alpha:
            selected_feature_ix[i] = False
        else:
            selected_feature_ix[i] = True
            indicator_feature_ls.append(feature)

    corr_selected_feature_names = feature_names[selected_feature_ix]
    X_train = X_train[:, selected_feature_ix]
    X_test = X_test[:, selected_feature_ix]

    return X_train, X_test, corr_selected_feature_names


def feature_selection_vif(X, feature_names, indicator_names, threshold=5.0):
    """
    Perform feature selection based on VIF.

    Parameters:
    X : ndarray or DataFrame
        The matrix of features.
    feature_names : list
        The list of feature names.
    threshold : float, optional
        The VIF threshold for removing features (default is 5.0).

    Returns:
    ndarray, list
        The reduced feature matrix and the list of selected feature names.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)

    while X.shape[1] > 1:

        if X.shape[1] >= X.shape[0]:
            fixed_ix = np.array([name not in indicator_names for name in feature_names])

            # Step 1: Correlation-based feature elimination
            corr_matrix = X.corr().abs() - np.identity(X.shape[1])
            strongest_corr_feature_ix = np.argmax(corr_matrix[~fixed_ix].max(1))
            to_drop = corr_matrix[~fixed_ix].iloc[strongest_corr_feature_ix].name
            X = X.drop(columns=[to_drop])
            feature_names = np.delete(feature_names, np.where([to_drop == feature_names])[1])
            print(to_drop, np.max(corr_matrix[~fixed_ix].max(1)))
        else:
            vif_data = calculate_vif(X.values, X.columns)
            fixed_ix = np.array([name not in indicator_names for name in feature_names])
            max_vif = vif_data[~fixed_ix].iloc[vif_data['vif'][~fixed_ix].argmax()]
            if max_vif.vif > threshold:
                X = X.drop(columns=[max_vif.feature])
                feature_names = np.delete(feature_names, max_vif.name)
                #print(max_vif.feature, max_vif.vif)
            else:
                break

    return X.values, X.columns.values
