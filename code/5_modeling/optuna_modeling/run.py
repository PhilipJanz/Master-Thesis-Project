import os
import pickle

from config import RUN_DIR


def list_of_runs():
    return os.listdir(RUN_DIR)


def open_run(run_name):
    # Open a run
    with open(RUN_DIR / f'{run_name}/run.pkl', 'rb') as file:
        run = pickle.load(file)
    return run


class Run:
    def __init__(self,
                 name,
                 cluster_set,
                 model_types,
                 opti_duration,
                 n_startup_trials):
        self.name = name
        self.cluster_set = cluster_set
        self.model_types = model_types
        self.opti_duration = opti_duration
        self.n_startup_trials = n_startup_trials

        # create folder structure
        self.create_folders()

        # save that run

    def create_folders(self):
        pass
