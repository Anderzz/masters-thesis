# file to define constants
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))

# Output directory for experiments:
EXPERIMENTS_DIR = os.path.join(this_file_path, "experiments")

DEFAULT_EVAL_CONFIG_LOC = os.path.join(
    this_file_path, "configs/eval/default_eval_config.yaml"
)

DEFAULT_PREPROCESSING_CONFIG_LOC = os.path.join(
    this_file_path, "configs/preprocessing/default_preprocessing_config.yaml"
)

DEFAULT_TRAINING_CONFIG_LOC = os.path.join(
    this_file_path, "configs/training/default_training_config.yaml"
)

DEFAULT_TESTING_CONFIG_LOC = os.path.join(
    this_file_path, "configs/testing/default_testing_config.yaml"
)

DEFAULT_TRAINING_CONFIG_LOC_HUNT4 = os.path.join(
    this_file_path, "configs/training/training_config_hunt4.yaml"
)

DEFAULT_TESTING_CONFIG_LOC_HUNT4 = os.path.join(
    this_file_path, "configs/testing/testing_config_hunt4.yaml"
)
