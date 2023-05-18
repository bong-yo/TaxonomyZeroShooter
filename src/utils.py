from os import stat
from typing import List, Any
import json
import openpyxl
import numpy as np
import pickle


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.divide(np.exp(x), np.sum(np.exp(x), axis=-1).reshape(-1, 1))

def normalize(x):
    """Bring cosine similarity into interval [0, 1], and normalize to 1"""
    x = x - np.min(x, axis=-1).reshape(-1, 1)
    return x / np.sum(x, axis=-1).reshape(-1, 1)

def cap_negs_to_zero(x):
    """Set to 0 all values less than 0"""
    x[x < 0] = 0.
    return x

def create_batches(data: List[Any], size: int) -> List[List[str]]:
    n_batches = len(data) // size + int(len(data) % size != 0)
    return [data[i * size: (i + 1) * size] for i in range(n_batches)]

def H(x):
    x = cap_negs_to_zero(x)
    x = x / np.sum(x, axis=-1).reshape(-1, 1)
    N = x.shape[-1]
    Hmax = np.log2(N)
    log_mask = np.ma.log(x)
    log_regularized = log_mask.filled(0)
    return -np.sum(x * log_regularized, axis=-1)  / Hmax

def entropy(x: np.array) -> np.array:
    n = x.shape[-1]
    return -np.sum(x * np.log2(x), axis=-1) / np.log2(n)

def STDev(x):
    return np.sqrt(np.sum((x - np.mean(x, axis=-1).reshape(-1, 1))**2, axis=-1)/x.shape[-1])


class FileIO:
    @staticmethod
    def read_text(filename):
        with open(filename, "r", encoding="utf8") as f:
            return f.read()

    @staticmethod
    def write_text(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            f.write(data)

    @staticmethod
    def append_text(data: str, filename):
        with open(filename, "a", encoding="utf8") as f:
            f.write("\n" + data)

    @staticmethod
    def read_json(filename):
        with open(filename, "r", encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def write_json(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            json.dump(data, f, default=str)

    @staticmethod
    def read_excel(filename, sheet_name="Sheet1"):
        wb_obj = openpyxl.load_workbook(filename)
        return wb_obj[sheet_name]

    @staticmethod
    def write_numpy(data, filename):
        with open(filename, 'wb') as f:
            np.save(f, data)

    @staticmethod
    def read_numpy(filename):
        with open(filename, 'rb') as f:
            return np.load(f)

    @staticmethod
    def write_pickle(data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def read_pickle(filename):
        with open(filename,'rb') as f:
            return pickle.load(f)
