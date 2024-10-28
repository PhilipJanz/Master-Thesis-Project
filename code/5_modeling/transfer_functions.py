import numpy as np
import pandas as pd


def get_last_hidden(model, X):
    # extract learned features for all data by taking results of last hidden layer
    transfer_feature_mtx = []
    # get last weights before network output
    last_weights = model.layers[-1].get_weights()
    # set all weights to zero
    last_weights[0] *= 0
    last_weights[1] *= 0
    # for each unit set weight to 1 so the networks output is the output of that hidden unit
    for i in range(len(last_weights[0])):
        last_weights[0] *= 0
        last_weights[0][i] = 1
        model.layers[-1].set_weights(last_weights)
        transfer_feature_mtx.append(model.predict(X, verbose=0))

    transfer_feature_df = pd.DataFrame(np.hstack(transfer_feature_mtx),
                                       columns=[f"transfer_feature_{i + 1}" for i in range(len(last_weights[0]))])
    return transfer_feature_df
