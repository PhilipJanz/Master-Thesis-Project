from copy import deepcopy

import keras
import numpy as np
import scikeras

import tensorflow as tf
from matplotlib import pyplot as plt

from models import create_nn


def train_and_predict(X_train, y_train, X_test, y_test, model, plot_train_history=None):
    """
    Train the model and predict the test set.
    This function is designed to run in parallel for each cross-validation fold.
    """
    if type(model) == scikeras.wrappers.KerasRegressor: # Keras NN
        my_model = create_nn(activation=model.model__activation,
                             dropout_rate=model.model__dropout_rate,
                             learning_rate=model.model__learning_rate,
                             neurons=model.model__neurons,
                             optimizer=model.model__optimizer,
                             input_shape=(X_train.shape[1],))

        if not plot_train_history:
            my_model.fit(X_train, y_train, epochs=model.epochs, batch_size=model.batch_size, verbose=0)
        else:
            history = my_model.fit(X_train, y_train, epochs=model.epochs, batch_size=model.batch_size,
                                   verbose=0, validation_data=(X_test, y_test))
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            plt.figure(figsize=(10, 6))
            plt.plot(loss, 'bo-', label='Training loss', alpha=0.7)
            plt.plot(val_loss, 'ro-', label='Validation loss', alpha=0.7)
            plt.hlines(xmin=0, xmax=len(loss), y=min(val_loss), color="darkgrey", label=f"Minimum val loss: {round(min(val_loss), 3)}, ({np.argmin(val_loss)}); Final val-loss: {round(val_loss[-1], 3)}")
            plt.title(f'Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.ylim([0.03, 5])  # Be cautious with log scale: 0 cannot be shown on a log scale
            plt.yscale('log')
            plt.legend()
            plt.show()
        y_pred = my_model.predict(X_test, verbose=0).T[0]
    else: # Sklearn
        my_model = deepcopy(model)
        my_model.fit(X_train, y_train)
        y_pred = my_model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    return y_pred, mse, my_model
