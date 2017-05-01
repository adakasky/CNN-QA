"""
char-based LSTM Encoder-Decoder model of question answering

@author: Ao Liu
"""

from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, BatchNormalization, LSTM, Bidirectional
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(0)


class LSTMCharModel(object):
    def __init__(self, char_dim=100, max_sent_len=100, max_doc_len=1000, n_chars=500):
        self.char_dim = char_dim
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.n_chars = n_chars

    def build(self):
        question_input = Input((self.max_sent_len,))
        passage_input = Input((self.max_doc_len,))
        question_embedding = Embedding(self.n_chars, self.char_dim, input_length=self.max_sent_len, mask_zero=True)(
            question_input)
        question_encoding = Bidirectional(LSTM(self.char_dim, activation='relu'))(question_embedding)
        question_encoding = BatchNormalization()(question_encoding)

        passage_embedding = Embedding(self.n_chars, self.char_dim, input_length=self.max_doc_len, mask_zero=True)(
            passage_input)
        passage_encoding = Bidirectional(LSTM(self.char_dim, activation='relu'))(passage_embedding)
        passage_encoding = BatchNormalization()(passage_encoding)

        interaction = add([question_encoding, passage_encoding])
        interaction = BatchNormalization()(interaction)

        softmax1 = Dense(self.max_doc_len, activation='softmax')(interaction)
        softmax2 = Dense(self.max_doc_len, activation='softmax')(interaction)

        self.model = Model([question_input, passage_input], [softmax1, softmax2])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.model.summary()

    def fit(self, x_train, y_train, batch_size=50, epochs=50, validation_data=None, validation_split=0., shuffle=True):
        earlystop_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        check_cb = ModelCheckpoint('../models/lstm_char.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                   verbose=0, save_best_only=True, mode='min')

        return self.model.fit(x_train, y_train, batch_size, epochs, 1, [check_cb], validation_split, validation_data,
                              shuffle)

    def predict(self, x, batch_size=50):
        return np.array(self.model.predict(x, batch_size)).argmax(-1)

    def evaluate(self, x, y, batch_size=50):
        return self.model.evaluate(x, y, batch_size=batch_size)

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def save(self, model_file="../models/lstm_char.hdf5"):
        self.model.save(model_file)

    def load(self, model_file="../models/lstm_char_best.hdf5"):
        self.model = load_model(model_file)
        self.model.summary()
