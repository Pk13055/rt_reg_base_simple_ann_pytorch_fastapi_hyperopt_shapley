import os
import sys
import time
import uuid
import warnings

import numpy as np
from hyperopt import fmin, hp, tpe
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

import algorithm.model_trainer as model_trainer
import algorithm.utils as utils

# get model configuration parameters
model_cfg = utils.get_model_config()


def get_hpt_space(hpt_specs):
    param_grid = {}
    for hp_obj in hpt_specs:
        if hp_obj["run_HPO"] == False:
            val = hp.choice(hp_obj["name"], [hp_obj["default"]])
        elif hp_obj["type"] == "categorical":
            val = hp.choice(hp_obj["name"], hp_obj["categorical_vals"])
        elif hp_obj["type"] == "int" and hp_obj["search_type"] == "uniform":
            val = hp.quniform(
                hp_obj["name"], hp_obj["range_low"], hp_obj["range_high"], 1
            )

        elif hp_obj["type"] == "int" and hp_obj["search_type"] == "log-uniform":
            val = hp.loguniform(
                hp_obj["name"],
                hp_obj["range_low"],
                hp_obj["range_high"],
            )

        elif hp_obj["type"] == "real" and hp_obj["search_type"] == "uniform":
            val = hp.quniform(
                hp_obj["name"], hp_obj["range_low"], hp_obj["range_high"], 1
            )

        elif hp_obj["type"] == "real" and hp_obj["search_type"] == "log-uniform":
            val = hp.loguniform(
                hp_obj["name"],
                hp_obj["range_low"],
                hp_obj["range_high"],
            )
        else:
            raise Exception(
                f"Error creating Hyper-Param Grid. \
                Undefined value type: {hp_obj['type']} or search_type: {hp_obj['search_type']}. \
                Verify hpt_params.json file."
            )
        param_grid.update({hp_obj["name"]: val})
    return param_grid


def get_default_hps(hpt_specs):
    default_hps = [hp["default"] for hp in hpt_specs]
    return default_hps


def load_best_hyperspace(results_path):
    results = [f for f in list(sorted(os.listdir(results_path))) if "json" in f]
    if len(results) == 0:
        return None
    best_result_name = results[0]
    best_result_file_path = os.path.join(results_path, best_result_name)
    return utils.get_json_file(best_result_file_path, "best_hpt_results")


def save_best_parameters(results_path, hyper_param_path):
    """Plot the best model found yet."""
    space_best_model = load_best_hyperspace(results_path)
    if space_best_model is None:
        print("No models yet. Continuing...")
        return
    print("Best model yet:", space_best_model["model_name"])

    # Important: you must save the best parameters to /opt/ml/model/model_config/hyperparameters.json during HPO for them to persist
    utils.save_json(
        os.path.join(hyper_param_path, "hyperparameters.json"),
        space_best_model["space"],
    )


def have_hyperparams_to_tune(hpt_specs):
    for hp_obj in hpt_specs:
        if hp_obj["run_HPO"] == True:
            return True
    return False


def clear_hp_results_dir(results_path):
    if os.path.exists(results_path):
        for f in os.listdir(results_path):
            os.remove(os.path.join(results_path, f))
    else:
        os.makedirs(results_path)


def tune_hyperparameters(
    data, data_schema, num_trials, hyper_param_path, hpt_results_path
):
    # read hpt_specs file
    hpt_specs = utils.get_hpt_specs()
    # check if any hyper-parameters are specified to be tuned
    if not have_hyperparams_to_tune(hpt_specs):
        print("No hyper-parameters to tune.")
        return

    print("Running HPT ...")

    start = time.time()

    # clear previous results, if any
    clear_hp_results_dir(hpt_results_path)

    # get the hpt space (grid) and default hps
    hpt_space = get_hpt_space(hpt_specs)
    default_hps = get_default_hps(hpt_specs)

    def objective(hyperparameters):
        """Build a model from this hyper parameter permutation and evaluate its performance"""

        # set random seeds
        utils.set_seeds()

        # perform train/valid split on the training data
        train_data, valid_data = train_test_split(
            data, test_size=model_cfg["valid_split"]
        )
        train_data, valid_data, _ = model_trainer.preprocess_data(
            train_data, valid_data, data_schema
        )
        train_X = train_data["X"].values.astype(np.float)
        train_y = train_data["y"].values.reshape(-1, 1).astype(np.float)
        valid_X = valid_data["X"].values.astype(np.float)
        valid_y = valid_data["y"].values.reshape(-1, 1).astype(np.float)

        # train model
        model = model_trainer.train_model(train_X, train_y, None, None, hyperparameters)

        # evaluate the model
        score: float = model.evaluate(valid_X, valid_y)

        # Our optimizing metric is the model loss fn
        opt_metric = score
        if np.isnan(opt_metric):
            opt_metric = 1.0e5  # sometimes loss becomes inf, so use a large value
        # create a unique model name for the trial - we add loss into file name
        # so we can later sort by file names, and get the best score file without reading each file
        model_name = f"model_{str(opt_metric)}_{str(uuid.uuid4())[:5]}"
        print("trial model:", model_name)
        # create trial result dict
        result = {
            "model_name": model_name,
            "space": hyperparameters,
            "loss": opt_metric,
            # 'history': pd.DataFrame(history.history).to_json(),
        }
        # Save training results to disks with unique filenames
        utils.save_json(os.path.join(hpt_results_path, model_name + ".json"), result)
        # Save the best model parameters found so far in case the HPO job is killed
        save_best_parameters(hpt_results_path, hyper_param_path)
        return -opt_metric

    n_initial_points = int(max(1, min(num_trials / 3, 5)))
    n_calls = max(
        2, num_trials
    )  # gp_minimize needs at least 2 trials, or it throws an error
    gp_ = fmin(
        objective,  # the objective function to minimize
        hpt_space,  # the hyperparameter space
        algo=tpe.suggest,
        max_evals=n_calls,  # the number of total evaluations of f(x), including n_initial_points
        rstate=np.random.default_rng(0),
    )

    end = time.time()
    print(f"Total HPO time: {np.round((end - start)/60.0, 2)} minutes")
