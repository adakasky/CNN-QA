"""
main script to train the model

@author: Ao Liu, Zhuodong Huang, Zitao Wang
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from cnn_char_model import CNNCharModel
from utils import load_marco, load_squad
from keras.utils.vis_utils import plot_model


# x, y = load_marco()

x, y, ids = load_squad()

x_train = [x[0][:6400], x[1][:6400]]
y_train = [y[0][:6400], y[1][:6400]]
x_dev = [x[0][6400:7200], x[1][6400:7200]]
y_dev = [y[0][6400:7200], y[1][6400:7200]]
x_test = [x[0][7200:8100], x[1][7200:8100]]
y_test = [y[0][7200:8100], y[1][7200:8100]]

cnn_char = CNNCharModel()
cnn_char.build()
# plot_model(cnn_char.model, to_file='../models/cnn_char_model.png')
cnn_char.fit([x[0][:20], x[1][:20]], [y[0][:20], y[1][:20]], 200, 200)

# cnn_char.save()
print(cnn_char.predict([x[0][:20], x[1][:20]]))
