import os
import pickle
import shutil

import numpy as np
import pandas as pd

from config import RUN_DIR, BASE_DIR


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
        self.run_dir = self.create_folders()

        # save copy of python code that created the run
        shutil.copy(BASE_DIR / "code/5_modeling/optuna_modeling/yield_prediction_optuna.py", self.run_dir / "yield_prediction_optuna.py")

        # save that run
        self.save()

    def create_folders(self):
        # run directory
        run_dir = RUN_DIR / self.name

        # make folders
        os.mkdir(run_dir)
        os.mkdir(run_dir / "models")
        os.mkdir(run_dir / "params")
        os.mkdir(run_dir / "plots")
        os.mkdir(run_dir / "plots/regional")
        os.mkdir(run_dir / "plots/overall")

        return run_dir

    def save(self):
        # Open a file in write mode
        with open(self.run_dir / 'run.pkl', 'wb') as file:
            # Serialize and save the object to the file
            pickle.dump(self, file)

    def save_predictions(self, prediction_df):
        assert "y_pred" in prediction_df.columns, "'y_pred' is not in the prediction-df."

        prediction_df.to_csv(self.run_dir / "prediction.csv", index=False)

    def save_performance(self, prediction_df, cluster_set):
        assert "y_pred" in prediction_df.columns, "'y_pred' is not in the prediction-df."

        # init performance dict that will become a dataframe
        performance_dict = {'country': [], 'adm1': [], 'adm2': [], cluster_set: [],
                            "avg_opt_trials": [], "avg_train_mse": [], "mse": [], "nse": []}

        # fill performance dict for each admin
        for adm, adm_prediction_df in prediction_df.groupby("adm"):
            for col in list({'country', 'adm1', 'adm2', cluster_set}):
                performance_dict[col].append(adm_prediction_df[col].iloc[0])

            performance_dict["avg_opt_trials"].append(np.mean(adm_prediction_df["n_opt_trials"]))
            performance_dict["avg_train_mse"].append(np.mean(adm_prediction_df["train_mse"]))
            mse = np.mean((adm_prediction_df["y_pred"] - adm_prediction_df["yield_anomaly"]) ** 2)
            performance_dict["mse"].append(mse)
            performance_dict["nse"].append(1 - mse / np.var(adm_prediction_df["yield_anomaly"]))

        # save it
        pd.DataFrame(performance_dict).to_csv(self.run_dir / "performance.csv", index=False)

    def load_prediction(self):
        return pd.read_csv(self.run_dir / "prediction.csv", keep_default_na=False)

    def save_model_and_params(self, name, model, params):
        # save params dict
        with open(self.run_dir / f"params/{name}.pkl", 'wb') as f:
            # Pickle using the highest protocol available.
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

        # save the model
        if params["model_type"] != "nn":
            with open(self.run_dir / f"models/{name}.pkl", 'wb') as f:
                # Pickle using the highest protocol available.
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
        else:
            model.save(self.run_dir / f"models/{name}.keras")
