import os
import json
import logging
import shutil

def get_config(config_path,config_model):
    if os.path.isfile(config_path):
        train_config = load_config(config_path)
    else:
        train_config = config_model()
        save_config(train_config, config_path)
    return train_config

def load_config(config_file):
    with open(config_file, encoding="utf8") as f:
        return json.load(f)

def save_config(config, config_file):
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def make_path(params):
    """
    Make folders for training and evaluation
    """
    if not os.path.isdir(params.checkpoints_path):
        os.makedirs(params.checkpoints_path)
    if not os.path.isdir(params.config_path):
        os.makedirs(params.config_path)
    if not os.path.isdir(params.log_path):
        os.makedirs(params.log_path)

def clean(params):
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)
    if os.path.isdir(params.config_path):
        shutil.rmtree(params.config_path)
    if os.path.isdir(params.checkpoints_path):
        shutil.rmtree(params.checkpoints_path)
    if os.path.isdir(params.log_path):
        shutil.rmtree(params.log_path)


