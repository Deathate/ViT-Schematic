import optuna

# import optuna.trial
from optuna.trial._trial import Trial

import config
from main import main


def objective(trial: Trial):
    config.DEPTH = trial.suggest_int("x1", 2, 6)
    config.NUM_HEADS = trial.suggest_int("x2", 1, 10)
    config.EMBED_DIM = trial.suggest_categorical("x3", [16, 32, 50, 64, 128])
    config.PATCH_SIZE = trial.suggest_categorical("x4", [5, 10, 20, 25])
    object = main()
    return object


study = optuna.create_study(storage="sqlite:///db.sqlite3")  # Create a new study with database.
study.optimize(objective, n_trials=100)
