"""
dilated CNN-CNN Encoder-Decoder model of question answering

@author: Ao Liu
"""

from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Embedding, BatchNormalization, Flatten, LSTM, Bidirectional
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K


class CNN(object):
    def __init__(self, word_dim=100, max_sent_len=50, max_doc_len=20, vocab_size=10000, max_cnn_layer=4,
                 model_file="../data/"):
        self.word_dim = word_dim
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.vocab_size = vocab_size
        self.max_cnn_layer = max_cnn_layer
        self.model_file = model_file

    def build(self):
        question_input = Input((self.max_sent_len,))
        passage_inputs = [Input((self.max_sent_len,)) for i in range(self.max_doc_len)]
        self.embedding_layer = Embedding(self.vocab_size, self.word_dim, input_length=self.max_sent_len)

        question_embedding = self.embedding_layer(question_input)
        passage_embeddings = []
        for passage_input in passage_inputs:
            passage_embeddings.append(self.embedding_layer(passage_input))

        encoders = [Conv1D(self.word_dim, 3, activation='relu', padding='same')]
        for i in range(2, self.max_cnn_layer + 1):
            encoders.append(Conv1D(self.word_dim, 3, activation='relu', dilation_rate=i, padding='same'))

        question_encoding = encoders[0](question_embedding)
        for encoder in encoders[1:]:
            question_encoding = encoder(question_encoding)
        question_encoding = BatchNormalization()(question_encoding)

        passage_encodings = []
        for embedding in passage_embeddings:
            encoding = encoders[0](embedding)
            for encoder in encoders[1:]:
                encoding = encoder(encoding)
            passage_encodings.append(encoding)
        passage_encoding = concatenate(passage_encodings)
        passage_encoding = Dense(self.word_dim, activation='relu')(passage_encoding)
        passage_encoding = BatchNormalization()(passage_encoding)

        merge_layer = concatenate([question_encoding, passage_encoding])
        decoder = Conv1D(self.word_dim, 3, activation='relu', padding='causal')(merge_layer)
        for i in range(2, self.max_cnn_layer + 1):
            decoder = Conv1D(self.word_dim, 3, activation='relu', padding='causal', dilation_rate=i)(decoder)

        attention = concatenate([question_encoding, decoder])
        lstm = Bidirectional(LSTM(self.max_sent_len, activation='relu', recurrent_activation='relu',
                                  dropout=0.1, recurrent_dropout=0.1, return_sequences=True))(attention)
        softmax = Dense(self.vocab_size + 1, activation='softmax')(lstm)
        # softmax = Flatten()(softmax)

        self.model = Model([question_input] + passage_inputs, softmax)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
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

    def save(self):
        self.model.save(self.model_file)
