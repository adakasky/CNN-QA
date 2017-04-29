"""
main script to train the model

@author: Ao Liu, Zhuodong Huang, Zitao Wang
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from cnn import CNN
from lstm_model import LSTMModel
from utils import load_data
from keras.utils.vis_utils import plot_model


x, y = load_data()

lstm = LSTMModel()
lstm.build()
# plot_model(lstm.model, to_file='../models/lstm_model.png')
lstm.fit(x, y, 100)
lstm.save()
print(lstm.predict(list(map(lambda a: a[:5], x))))
