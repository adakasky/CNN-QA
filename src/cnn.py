"""
dilated CNN-CNN Encoder-Decoder model of question answering

@author: Ao Liu
"""

from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np

from keras import losses
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Embedding, BatchNormalization, Flatten, LSTM, Bidirectional, Masking, MaxPool1D
from keras.layers.merge import concatenate, add
from keras.callbacks import ModelCheckpoint, EarlyStopping


class CNN(object):
    def __init__(self, word_dim=100, max_sent_len=50, max_doc_len=20, vocab_size=10000, max_cnn_layer=5):
        self.word_dim = word_dim
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.vocab_size = vocab_size
        self.max_cnn_layer = max_cnn_layer

    def build(self):
        question_input = Input((self.max_sent_len,))
        passage_inputs = [Input((self.max_sent_len,)) for i in range(self.max_doc_len)]
        question_embedding = Embedding(self.vocab_size + 1, self.word_dim, input_length=self.max_sent_len)(
            question_input)
        question_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same')(question_embedding)
        question_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=2)(
            question_encoding)
        question_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=4)(
            question_encoding)
        question_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=8)(
            question_encoding)
        question_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=16)(
            question_encoding)

        passage_embedding_layer = Embedding(self.vocab_size + 1, self.word_dim, input_length=self.max_sent_len)
        passage_embeddings = []
        for passage_input in passage_inputs:
            passage_embeddings.append(passage_embedding_layer(passage_input))
        passage_embedding = concatenate(passage_embeddings, 1)
        passage_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same')(passage_embedding)
        passage_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=2)(
            passage_encoding)
        passage_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=4)(
            passage_encoding)
        passage_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=8)(
            passage_encoding)
        passage_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=16)(
            passage_encoding)
        passage_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=32)(
            passage_encoding)
        passage_encoding = Conv1D(self.word_dim, 3, activation='relu', padding='same', dilation_rate=64)(
            passage_encoding)

        interaction = concatenate([question_encoding, passage_encoding], 1)
        interaction = MaxPool1D(self.max_doc_len + 1)(interaction)
        decoding = Conv1D(self.word_dim, 3, activation='relu', padding='causal')(interaction)
        decoding = Conv1D(self.word_dim, 3, activation='relu', padding='causal', dilation_rate=2)(decoding)
        decoding = Conv1D(self.word_dim, 3, activation='relu', padding='causal', dilation_rate=4)(decoding)
        decoding = Conv1D(self.word_dim, 3, activation='relu', padding='causal', dilation_rate=8)(decoding)
        decoding = Conv1D(self.word_dim, 3, activation='relu', padding='causal', dilation_rate=16)(decoding)
        decoding = Conv1D(self.word_dim, 3, activation='relu', padding='causal', dilation_rate=32)(decoding)
        decoding = Conv1D(self.word_dim, 3, activation='relu', padding='causal', dilation_rate=64)(decoding)

        attention = concatenate([question_encoding, decoding], 1)
        attention = MaxPool1D(2)(attention)
        softmax = Dense(self.vocab_size + 1, activation='softmax')(attention)

        def masked_sparse_categorical_crossentropy(y_true, y_pred):
            mask = np.ones((self.max_sent_len, self.vocab_size + 1))
            mask[:, 0] = 0
            mask[:, -1] = 0
            mask = K.constant(mask)
            return K.sparse_categorical_crossentropy(y_pred * mask, y_true)

        self.model = Model([question_input] + passage_inputs, softmax)
        self.model.compile(loss=masked_sparse_categorical_crossentropy, optimizer='adadelta', metrics=['accuracy'])
        self.model.summary()

    def fit(self, x_train, y_train, epochs=50, batch_size=50, validation_data=None, validation_split=0.1, shuffle=True):
        earlystop_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
        check_cb = ModelCheckpoint('../checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                                   verbose=0, save_best_only=True, mode='min')

        return self.model.fit(x_train, y_train, batch_size, epochs, 1, [check_cb], validation_split,
                              shuffle=shuffle)

    def predict(self, x, batch_size=50):
        return np.array(self.model.predict(x, batch_size)).argmax(-1)

    def evaluate(self, x, y, batch_size=50):
        return self.model.evaluate(x, y, batch_size=batch_size)

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def save(self, model_file="../data/"):
        self.model.save(model_file)
