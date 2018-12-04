# -*- coding:utf-8 -*-
#文本数据读取及预处理
import re
import numpy as np
from tensorflow.contrib import learn
import math
import os
import shutil
import logging
import json

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(data_pos_path,data_neg_path):
    pos_lines = open(data_pos_path, 'r', encoding='utf8').readlines()
    neg_lines = open(data_neg_path, 'r', encoding='utf8').readlines()
    #清洗数据
    pos_data = [clean_str(line).strip() for line in pos_lines]
    neg_data = [clean_str(line).strip() for line in neg_lines]
    x_data = pos_data + neg_data
    #pos:[0,1],neg:[1,0]
    y_pos = [[0,1] for _ in pos_lines]
    y_neg = [[1,0] for _ in neg_lines]
    y_data = np.concatenate([y_pos,y_neg],0)

    return x_data,y_data

def char_mapping(seq_max,x_data,vec_path):
    # 将每个词生成一个字典，并按照最大维度填充0
    voca_processor = learn.preprocessing.VocabularyProcessor(seq_max)
    x_data = np.array(list(voca_processor.fit_transform(x_data)))
    voca_processor.save(vec_path)
    return x_data,voca_processor

#将数据打乱
def shuffledData(x_data,y_data):
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_data)))
    x_shuffled = x_data[shuffle_indices]
    y_shuffled = y_data[shuffle_indices]
    return x_shuffled,y_shuffled

def getDivideData(x_data,y_data):
    train_num = math.ceil(0.8 * len(y_data))
    return x_data[:train_num],y_data[:train_num],x_data[train_num:],y_data[train_num:]

def iter_data(x_data,y_data,batch_size):
    num_batch = (len(y_data) // batch_size) + 1
    x_data, y_data = shuffledData(x_data, y_data)
    for batch_index in range(num_batch):
        yield x_data[batch_index:(batch_index + 1) * batch_size],y_data[batch_index:(batch_index + 1) * batch_size]


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
    if os.path.isfile(params.vec_file):
        os.remove(params.vec_file)
    if os.path.isdir(params.config_path):
        shutil.rmtree(params.config_path)
    if os.path.isdir(params.checkpoints_path):
        shutil.rmtree(params.checkpoints_path)
    if os.path.isdir(params.log_path):
        shutil.rmtree(params.log_path)











