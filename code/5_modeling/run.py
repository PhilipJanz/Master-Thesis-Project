import os
import pickle
import joblib
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
                 objective,
                 cluster_set,
                 model_types,
                 timeout,
                 n_trials,
                 n_startup_trials,
                 python_file):
        self.name = name
        self.objective = objective
        self.cluster_set = cluster_set
        self.model_types = model_types
        self.timeout = timeout
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials

        # create folder structure
        self.run_dir = self.create_folders()

        # save copy of python code that created the run
        shutil.copy(python_file, self.run_dir / str(python_file).split("\\")[-1])

        # save that run
        self.save()

    def create_folders(self):
        # run directory
        run_dir = RUN_DIR / self.name

        # make folders
        os.mkdir(run_dir)
        os.mkdir(run_dir / "models")
        os.mkdir(run_dir / "params")
        os.mkdir(run_dir / "optuna_studies")
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
        if cluster_set == "all":
            performance_dict = {'country': [], 'adm1': [], 'adm2': [],
                                "avg_opt_trials": [], "avg_train_mse": [], "mse": [], "nse": []}
        else:
            performance_dict = {'country': [], 'adm1': [], 'adm2': [], cluster_set: [],
                                "avg_opt_trials": [], "avg_train_mse": [], "mse": [], "nse": []}

        # fill performance dict for each admin
        for adm, adm_prediction_df in prediction_df.groupby("adm"):
            for col in list({'country', 'adm1', 'adm2', cluster_set}):
                if col == "all":
                    continue
                performance_dict[col].append(adm_prediction_df[col].iloc[0])

            performance_dict["avg_opt_trials"].append(np.mean(adm_prediction_df["n_opt_trials"]))
            performance_dict["avg_train_mse"].append(np.mean(adm_prediction_df["train_mse"]))
            mse = np.mean((adm_prediction_df["y_pred"] - adm_prediction_df[self.objective]) ** 2)
            performance_dict["mse"].append(mse)
            performance_dict["nse"].append(1 - mse / np.var(adm_prediction_df[self.objective]))

        # save it
        pd.DataFrame(performance_dict).to_csv(self.run_dir / "performance.csv", index=False)

    def save_optuna_study(self, study):
        study_path = self.run_dir / f"optuna_studies/{study.study_name}.pkl"
        # save it
        joblib.dump(study, study_path)
        return None

    def try_load_optuna_study(self, study_name):
        existing_studies = os.listdir(self.run_dir / f"optuna_studies")
        if study_name in [file_name.replace(".pkl", "") for file_name in existing_studies]:
            return joblib.load(self.run_dir / f"optuna_studies/{study_name}.pkl")
        else:
            return None


    def load_prediction(self):
        return pd.read_csv(self.run_dir / "prediction.csv", keep_default_na=False)

    def load_model_and_params(self):
        file_names = os.listdir(self.run_dir / "params")
        model_dir = {}
        params_dir = {}
        for file_name in file_names:
            with open(self.run_dir / f'models/{file_name}', 'rb') as file:
                model_dir[file_name] = pickle.load(file)
            with open(self.run_dir / f'params/{file_name}', 'rb') as file:
                params = pickle.load(file)
                if not params_dir:
                    params_dir = {k: [v] for k, v in params.items()}
                    params_dir["cluster_name"] = [file_name[:-9]]
                    params_dir["year"] = [int(file_name[-8:-4])]
                else:
                    for k, v in params.items():
                        params_dir[k].append(v)
                    params_dir["cluster_name"].append(file_name[:-9])
                    params_dir["year"].append(int(file_name[-8:-4]))
        feature_ls_ls = params_dir["feature_names"]
        del params_dir["feature_names"]
        params_df = pd.DataFrame(params_dir)

        result_df = self.load_prediction()
        result_df = result_df[result_df["y_pred"] != ""]
        result_df["y_pred"] = pd.to_numeric(result_df["y_pred"])

        performance_dict = {"cluster_name": [], "year": [], "mse": [], "nse": []}
        for adm_year, adm_year_results_df in result_df.groupby(["adm1_", "harv_year"]):

            mse = np.mean((adm_year_results_df["y_pred"] - adm_year_results_df[self.objective]) ** 2)

            nse = 1 - mse / np.mean(adm_year_results_df[self.objective] ** 2)

            # fill dict
            performance_dict["cluster_name"].append(adm_year[0])
            performance_dict["year"].append(adm_year[1])
            performance_dict["mse"].append(mse)
            performance_dict["nse"].append(nse)

        performance_df = pd.DataFrame(performance_dict)

        return model_dir, params_df, feature_ls_ls

    def load_params(self):
        file_names = os.listdir(self.run_dir / "params")
        params_dir = {}
        for file_name in file_names:
            with open(self.run_dir / f'params/{file_name}', 'rb') as file:
                params = pickle.load(file)
                if not params_dir:
                    params_dir = {k: [v] for k, v in params.items()}
                    params_dir["cluster_name"] = [file_name[:-9]]
                    params_dir["year"] = [int(file_name[-8:-4])]
                else:
                    for k, v in params.items():
                        params_dir[k].append(v)
                    params_dir["cluster_name"].append(file_name[:-9])
                    params_dir["year"].append(int(file_name[-8:-4]))
        feature_ls_ls = params_dir["feature_names"]
        del params_dir["feature_names"]
        params_df = pd.DataFrame(params_dir)

        result_df = self.load_prediction()
        result_df = result_df[result_df["y_pred"] != ""]
        result_df["y_pred"] = pd.to_numeric(result_df["y_pred"])

        performance_dict = {"cluster_name": [], "year": [], "mse": [], "nse": []}
        for adm_year, adm_year_results_df in result_df.groupby(["adm1_", "harv_year"]):

            mse = np.mean((adm_year_results_df["y_pred"] - adm_year_results_df["yield_anomaly"]) ** 2)

            nse = 1 - mse / np.mean(adm_year_results_df["yield_anomaly"] ** 2)

            # fill dict
            performance_dict["cluster_name"].append(adm_year[0])
            performance_dict["year"].append(adm_year[1])
            performance_dict["mse"].append(mse)
            performance_dict["nse"].append(nse)

        performance_df = pd.DataFrame(performance_dict)

        return params_df, feature_ls_ls

    def save_model_and_params(self, name, model, params, model_type):
        # save params dict
        with open(self.run_dir / f"params/{name}.pkl", 'wb') as f:
            # Pickle using the highest protocol available.
            pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

        # save the model
        if model_type in ["nn", "lstm"]:
            model.save(self.run_dir / f"models/{name}.keras")
        else:
            with open(self.run_dir / f"models/{name}.pkl", 'wb') as f:
                # Pickle using the highest protocol available.
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
