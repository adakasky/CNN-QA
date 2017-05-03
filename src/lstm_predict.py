"""
main script to predict using trained the model

@author: Ao Liu, Zhuodong Huang, Zitao Wang
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from lstm_char_model import LSTMCharModel
from utils import load_squad, get_prediction
from rouge import Rouge


x, y, ids = load_squad()

x_train = [x[0][:6400], x[1][:6400]]
y_train = [y[0][:6400], y[1][:6400]]
id_train = ids[:6400]
x_dev = [x[0][6400:7200], x[1][6400:7200]]
y_dev = [y[0][6400:7200], y[1][6400:7200]]
id_dev = ids[6400:7200]
x_test = [x[0][7200:8100], x[1][7200:8100]]
y_test = [y[0][7200:8100], y[1][7200:8100]]
ids_test = ids[7200:8100]


lstm = LSTMCharModel()
lstm.load()

y_truth, y_pred = get_prediction(lstm, x_test, y_test)

