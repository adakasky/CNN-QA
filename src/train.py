"""
main script to train the model

@author: Ao Liu, Zhuodong Huang, Zitao Wang
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from cnn import CNN
from utils import load_data
from keras.utils.vis_utils import plot_model


cnn = CNN()
cnn.build()
# plot_model(cnn.model, to_file='../checkpoints/model.png')
x, y = load_data()
# print(x[0].shape, y.shape)
cnn.fit(x, y)
# outputs = [layer.get_output_at(-1) for layer in cnn.model.layers]
print(cnn.predict(list(map(lambda a: a[:5], x))))
