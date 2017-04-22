from __future__ import division
from __future__ import print_function

import numpy as np
from cnn import CNN
from utils import load_data
from keras.utils.vis_utils import plot_model

# load_data("../data/dev_v1.1.json.gz")
cnn = CNN()
cnn.build()
plot_model(cnn.model, to_file='../checkpoints/model.png')
x1 = np.array([np.ones(30), np.zeros(30)])
x2 = np.array([[np.zeros(30) for i in range(10)], [np.ones(30) for i in range(10)]])
y = [1, 0]
cnn.train_on_batch(x1, x2, y)
