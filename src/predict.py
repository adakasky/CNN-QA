"""
main script to predict using trained the model

@author: Ao Liu, Zhuodong Huang, Zitao Wang
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from cnn import CNN
from lstm_model import LSTMModel
from utils import load_data


x, y = load_data()

lstm = LSTMModel()
lstm.load()

print(lstm.predict(x))
