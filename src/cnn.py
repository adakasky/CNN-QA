"""
dilated CNN-CNN Encoder-Decoder model of question answering

@author: Ao Liu
"""

from __future__ import division
from __future__ import print_function
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Embedding, BatchNormalization
from keras.layers.merge import concatenate


class CNN(object):
    def __init__(self, word_dim=300, max_sent_len=30, max_doc_len=10, vocab_size=10000, max_cnn_layer=4):
        self.word_dim = word_dim
        self.max_sent_len = max_sent_len
        self.max_doc_len = max_doc_len
        self.vocab_size = vocab_size
        self.max_cnn_layer = max_cnn_layer

    def build(self):
        question_input = Input((self.max_sent_len,))
        passage_inputs = [Input((self.max_sent_len,)) for i in range(self.max_doc_len)]
        self.embedding_layer = Embedding(self.vocab_size, self.word_dim, input_length=self.max_sent_len)

        question_embedding = self.embedding_layer(question_input)
        passage_embeddings = []
        for passage_input in passage_inputs:
            passage_embeddings.append(self.embedding_layer(passage_input))

        encoders = [Conv1D(self.max_sent_len, 3, activation='relu')]
        for i in range(2, self.max_cnn_layer + 1):
            encoders.append(Conv1D(self.max_sent_len, 3, activation='relu', dilation_rate=i))

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
        passage_encoding = Dense(self.max_doc_len, activation='relu')(passage_encoding)
        passage_encoding = BatchNormalization()(passage_encoding)

        merge_layer = concatenate([question_encoding, passage_encoding])
        decoder = Conv1D(self.max_sent_len, 3, activation='relu', padding='causal')(merge_layer)
        for i in range(2, self.max_cnn_layer + 1):
            decoder = Conv1D(self.max_sent_len, 3, activation='relu', padding='causal', dilation_rate=i)(decoder)

        self.model = Model([question_input] + passage_inputs, decoder)
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.model.summary()

    def compile(self):
        pass

    def fit(self, x_question, x_passage, y, nb_epoch=50, batch_size=50, shuffle=True):
        return self.model.fit(np.append(x_question, x_passage, 1), y, nb_epoch=nb_epoch, batch_size=batch_size,
                              shuffle=shuffle)

    def evaluate(self, x_question, x_passage, y, batch_size=50):
        return self.model.evaluate(np.append(x_question, x_passage, 1), y, batch_size=batch_size)

    def train_on_batch(self, x_question, x_passage, y):
        return self.model.train_on_batch(np.append(x_question, x_passage, 1), y)

    def save(self):
        pass
