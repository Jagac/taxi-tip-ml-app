import json
import random

import numpy as np


def set_seeds(seed=42):
    np.random.seed()
    random.seed(seed)


def load_dict(filepath):
    with open(filepath, "r") as file:
        dictionary = json.load(file)

    return dictionary
