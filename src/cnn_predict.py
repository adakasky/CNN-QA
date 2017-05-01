"""
main script to predict using trained the model

@author: Ao Liu, Zhuodong Huang, Zitao Wang
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from cnn import CNN
from lstm_char_model import LSTMCharModel
from utils import load_data


x, y = load_data()

lstm_char = LSTMCharModel()
lstm_char.load()

print(lstm_char.predict(x))
