"""
LSTM Encoder-Decoder model of question answering

@author: Ao Liu
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from keras import losses
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, BatchNormalization, LSTM, Bidirectional, Reshape, Masking
from keras.layers.merge import concatenate, add, multiply
from keras.callbacks import ModelCheckpoint, EarlyStopping


np.random.seed(0)


class LSTMModel(object):
    def __init__(self, word_dim=100, max_sent_len=20, max_doc_len=100, vocab_size=10000, max_cnn_layer=5):
        self.word_dim = word_dim
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.vocab_size = vocab_size
        self.max_cnn_layer = max_cnn_layer

    def build(self):
        question_input = Input((self.max_sent_len,))
        passage_input = Input((self.max_doc_len,))
        question_embedding = Embedding(self.vocab_size, self.word_dim, input_length=self.max_sent_len,
                                       mask_zero=True)(question_input)
        question_encoding = Bidirectional(LSTM(self.word_dim, activation='relu'))(
            question_embedding)
        question_encoding = BatchNormalization()(question_encoding)

        passage_embedding = Embedding(self.vocab_size, self.word_dim, input_length=self.max_doc_len,
                                      mask_zero=True)(passage_input)
        passage_encoding = Bidirectional(LSTM(self.word_dim, activation='relu'))(
            passage_embedding)
        passage_encoding = BatchNormalization()(passage_encoding)

        interaction = concatenate([question_encoding, passage_encoding])
        interaction = BatchNormalization()(interaction)
        interaction = Dense(self.max_sent_len, activation='relu')(interaction)
        interaction = Reshape((self.max_sent_len, 1))(interaction)

        softmax = Dense(self.vocab_size + 1, activation='softmax')(interaction)

        def masked_sparse_categorical_crossentropy(y_true, y_pred):
            mask = np.ones((self.max_sent_len, self.vocab_size + 1))
            mask[:, 0] = 0
            mask[:, -1] = 0
            mask = K.constant(mask)
            return K.sparse_categorical_crossentropy(y_pred * mask, y_true)

        self.model = Model([question_input, passage_input], softmax)
        self.model.compile(loss=masked_sparse_categorical_crossentropy, optimizer='adadelta', metrics=['accuracy'])
        self.model.summary()

    def fit(self, x_train, y_train, epochs=50, batch_size=50, validation_data=None, validation_split=0.1, shuffle=True):
        earlystop_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        check_cb = ModelCheckpoint('../models/lstm.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                   verbose=0, save_best_only=True, mode='min')

        return self.model.fit(x_train, y_train, batch_size, epochs, 1, [check_cb], validation_split, shuffle=shuffle)

    def predict(self, x, batch_size=50):
        return np.array(self.model.predict(x, batch_size)).argmax(-1)

    def evaluate(self, x, y, batch_size=50):
        return self.model.evaluate(x, y, batch_size=batch_size)

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def save(self, model_file="../models/lstm.hdf5"):
        self.model.save(model_file)

    def load(self, model_file="../models/lstm.hdf5"):
        self.model = load_model(model_file)
        self.model.summary()
