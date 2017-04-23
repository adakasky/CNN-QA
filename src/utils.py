"""
utility functions

@author: Ao Liu, Zhuodong Huang, Zitao Wang
"""

from __future__ import division
from __future__ import print_function

import json
import gzip
import codecs
import numpy as np

from nltk import word_tokenize as wt
from nltk import sent_tokenize as st


# from gensim.models import Word2Vec
# embeddings = Word2Vec.load("../data/word2vec.bin")


def preprocess(input_file="../data/dev_v1.1.json.gz", vocab_file="../data/vocab.json", data_file="../data/data.txt",
               max_sent_len=50, max_doc_len=20, vocab_size=10000):
    lines = gzip.open(input_file, 'r').readlines()
    vocab_writer = codecs.open(vocab_file, 'w', "utf-8")
    data_writer = codecs.open(data_file, "w", "utf-8")

    vocab = {}
    for line in lines:
        content = json.loads(line.decode("utf-8"))

        query = content["query"]
        passages = content["passages"]
        answers = content["answers"]

        query_token = wt(query)
        query_token = list(map(lambda t: t.lower(), query_token))
        for token in query_token[:max_sent_len]:
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1
        data_writer.write(' '.join(query_token) + '\n')

        for passage in passages:
            sentences = st(passage["passage_text"])[:max_doc_len]
            for sentence in sentences:
                tokens = wt(sentence)
                tokens = list(map(lambda t: t.lower(), tokens))
                for token in tokens[:max_sent_len]:
                    if token not in vocab:
                        vocab[token] = 0
                    vocab[token] += 1
                if passage["is_selected"] and len(answers) == 1:
                    data_writer.write(' '.join(tokens) + '\n')

        answer_token = []
        for answer in answers:
            answer_token = wt(answer)
            answer_token = list(map(lambda t: t.lower(), answer_token))
            for token in answer_token[:max_sent_len]:
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
        if len(answers) == 1:
            data_writer.write(' '.join(answer_token) + '\n\n')

    sorted_count = sorted(vocab.items(), key=lambda t: t[1], reverse=True)
    sorted_count = list(map(lambda t: t[0], sorted_count[:vocab_size - 1]))
    json_out = {"index_to_token": {i + 1: t for i, t in enumerate(sorted_count)},
                "token_to_index": {t: i + 1 for i, t in enumerate(sorted_count)}}

    json_out["index_to_token"][vocab_size] = "UNK"
    json_out["token_to_index"]["UNK"] = vocab_size

    json.dump(json_out, vocab_writer)
    vocab_writer.flush()
    data_writer.flush()
    vocab_writer.close()
    data_writer.close()


def prepare_data(vocab_file="../data/vocab.json", data_file="../data/data.txt", output_file="../data/prepared_data.txt",
                 max_sent_len=50, max_doc_len=20):
    vocab = codecs.open(vocab_file, 'r', "utf-8")
    vocab = json.load(vocab)
    wtoi = vocab["token_to_index"]

    data = codecs.open(data_file, 'r', "utf-8").read().split("\n\n")
    writer = codecs.open(output_file, 'w', "utf-8")

    def write_sent(tokens):
        def write_token(t):
            if t in wtoi:
                writer.write("%d " % wtoi[t])
            else:
                writer.write("10000 ")

        for token in tokens[:-1]:
            write_token(token)
        if len(tokens) < max_sent_len:
            write_token(tokens[-1])
            for i in range(len(tokens) + 1, max_sent_len):
                writer.write("0 ")
            writer.write("0\n")
        else:
            if tokens[-1] in wtoi:
                writer.write("%d\n" % wtoi[tokens[-1]])
            else:
                writer.write("10000\n")

    for content in data:
        if content == "":
            continue
        lines = content.split("\n")
        query_token = lines[0].split()
        write_sent(query_token[:max_sent_len])

        sentences = lines[1:-1][:max_doc_len]
        for sentence in sentences:
            tokens = sentence.split()
            write_sent(tokens[:max_sent_len])
        if len(sentences) < max_sent_len:
            for i in range(len(sentences), max_doc_len):
                writer.write("0 " * 49)
                writer.write("0\n")

        answer_token = lines[-1].split()
        write_sent(answer_token[:max_sent_len])
        writer.write("\n")

    writer.flush()
    writer.close()


def load_data(file="../data/prepared_data.txt", max_sent_len=50, max_doc_len=20, vocab_size=10000):
    data = codecs.open(file, 'r', "utf-8").read().split("\n\n")

    x = [[] for i in range(max_doc_len + 1)]
    y = []
    for k, content in enumerate(data):
        if content == "":
            continue
        lines = content.split("\n")
        for i, line in enumerate(lines[:-1]):
            x[i].append([int(j) for j in line.split()])
        y.append([[int(i)] for i in lines[-1].split()])

    return list(np.array(x)), np.array(y)
