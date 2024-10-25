import os

import joblib
import optuna


def get_study_runtime_and_trials(file_path):
    # Load the study depending on the file type (.pkl or .db)
    if file_path.endswith('.pkl'):
        # Load study from a pickle file
        study = joblib.load(file_path)
    elif file_path.endswith('.db'):
        # Load study from a database file
        storage_url = f"sqlite:///{file_path}"
        study = optuna.load_study(study_name='default_study', storage=storage_url)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    trials = study.get_trials(deepcopy=False)

    if not trials:
        return 0, 0

    start_time = trials[0].datetime_start
    end_time = trials[-1].datetime_start

    # Filter out only completed or pruned trials
    num_trials = len(trials)

    if start_time and end_time:
        runtime = end_time - start_time
        return runtime.total_seconds(), num_trials  # Return runtime in seconds and number of valid trials
    else:
        return 0, num_trials


def calculate_total_runtime_and_average_trials(folder_path):
    total_runtime_seconds = 0
    total_trials = 0
    study_count = 0

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        runtime_seconds, num_trials = get_study_runtime_and_trials(file_path)
        total_runtime_seconds += runtime_seconds
        total_trials += num_trials
        study_count += 1

    total_runtime_hours = total_runtime_seconds / 3600  # Convert seconds to hours
    average_trials = total_trials / study_count if study_count > 0 else 0  # Avoid division by zero
    return total_runtime_hours, average_trials
