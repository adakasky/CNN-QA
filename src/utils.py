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


def preprocess_marco(input_file="../data/dev_v1.1.json.gz", vocab_file="../data/vocab_marco.json",
                     data_file="../data/marco_dev.txt", max_sent_len=20, max_doc_len=100, vocab_size=10000):
    lines = gzip.open(input_file, 'r').readlines()
    vocab_writer = codecs.open(vocab_file, 'w', "utf-8")
    data_writer = codecs.open(data_file, "w", "utf-8")
    vocab = {}
    for line in lines:
        content = json.loads(line.decode("utf-8"))

        query = content["query"]
        passages = content["passages"]
        answers = content["answers"]

        query_token = wt(query.lower())
        query_token = list(map(lambda t: t.lower(), query_token))
        for token in query_token:
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1
        if len(answers) == 1:
            data_writer.write(' '.join(query_token[:max_sent_len]) + '\t')

        for passage in passages:
            tokens = wt(passage["passage_text"].lower())

            for token in tokens:
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
            if passage["is_selected"] and len(answers) == 1:
                data_writer.write(' '.join(tokens[:max_doc_len]) + '\t')

        for answer in answers:
            answer_token = wt(answer.lower())

            for token in answer_token:
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1
            if len(answers) == 1:
                data_writer.write(' '.join(answer_token[:max_sent_len]) + '\n')

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


def prepare_marco(vocab_file="../data/vocab_marco.json", data_file="../data/marco_dev.txt",
                  output_file="../data/prepared_marco_dev.txt", max_sent_len=20, max_doc_len=100):
    vocab = codecs.open(vocab_file, 'r', "utf-8")
    vocab = json.load(vocab)
    wtoi = vocab["token_to_index"]

    data = codecs.open(data_file, 'r', "utf-8").read().split("\n")
    writer = codecs.open(output_file, 'w', "utf-8")

    def write_sent(tokens, max_len):
        def write_token(t):
            if t in wtoi:
                writer.write("%d " % wtoi[t])
            else:
                writer.write("10000 ")

        for token in tokens[:-1]:
            write_token(token)
        if len(tokens) < max_len:
            write_token(tokens[-1])
            for i in range(len(tokens) + 1, max_len):
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
        piece = content.split("\t")
        query_token = piece[0].split()
        write_sent(query_token, max_sent_len)

        passage_token = piece[1].split()
        write_sent(passage_token, max_doc_len)

        answer_token = piece[-1].split()
        write_sent(answer_token, max_sent_len)
        writer.write("\n")

    writer.flush()
    writer.close()


def load_marco(file="../data/prepared_marco_dev.txt"):
    data = codecs.open(file, 'r', "utf-8").read().split("\n\n")

    x = [[], []]
    y = []
    for content in data:
        if content == "":
            continue
        lines = content.split("\n")
        x[0].append([int(j) for j in lines[0].split()])
        x[1].append([int(j) for j in lines[1].split()])
        y.append([[int(i)] for i in lines[-1].split()])

    return [np.array(x[0]), np.array(x[1])], np.array(y)


def preprocess_squad(input_file="../data/dev-v1.1.json", char_file="../data/char_squad_dev.json",
                     data_file="../data/squad_dev.txt", max_sent_len=100, max_doc_len=1000):

    reader = codecs.open(input_file, 'r', "utf-8")
    data = json.load(reader)["data"]
    char_writer = codecs.open(char_file, 'w', "utf-8")
    data_writer = codecs.open(data_file, "w", "utf-8")
    chars = {}

    for content in data:
        paragraphs = content["paragraphs"]
        for paragraph in paragraphs:
            context = paragraph["context"]
            for c in context:
                if c not in chars:
                    chars[c] = 0
                chars[c] += 1
            qas = paragraph["qas"]
            for qa in qas:
                question = qa["question"]
                for c in question:
                    if c not in chars:
                        chars[c] = 0
                    chars[c] += 1
                answer_start = qa["answers"][0]["answer_start"]
                answer_end = answer_start + len(qa["answers"][0]["text"])
                qid = qa["id"]
                if len(context) < max_doc_len and len(question) < max_sent_len:
                    data_writer.write("%s\n%s\n%d\t%d\n%s\n\n" % (question, context, answer_start, answer_end, qid))

    sorted_count = sorted(chars.items(), key=lambda t: t[1], reverse=True)
    sorted_count = list(map(lambda t: t[0], sorted_count))
    json_out = {"index_to_char": {i + 1: t for i, t in enumerate(sorted_count)},
                "char_to_index": {t: i + 1 for i, t in enumerate(sorted_count)}}

    json.dump(json_out, char_writer)
    char_writer.flush()
    char_writer.close()
    data_writer.flush()
    data_writer.close()


def prepare_squad(char_file="../data/char_squad_dev.json", data_file="../data/squad_dev.txt",
                  output_file="../data/prepared_squad_dev.txt", max_sent_len=100, max_doc_len=1000):
    chars = codecs.open(char_file, 'r', "utf-8")
    chars = json.load(chars)
    ctoi = chars["char_to_index"]

    data = codecs.open(data_file, 'r', "utf-8").read().split("\n\n")
    writer = codecs.open(output_file, 'w', "utf-8")

    def write_chars(text, max_len):
        for c in text[:-1]:
            writer.write("%d " % ctoi[c])
        if len(text) < max_len:
            writer.write("%d " % ctoi[text[-1]])
            for i in range(len(text) + 1, max_len):
                writer.write("0 ")
            writer.write("0\n")
        else:
            writer.write("%d\n" % ctoi[text[-1]])

    for content in data:
        if content == "":
            continue
        piece = content.split("\n")
        question = piece[0]
        context = piece[1]
        offset = piece[2]
        qid = piece[3]
        write_chars(question, max_sent_len)
        write_chars(context, max_doc_len)
        writer.write("%s\n%s\n\n" % (offset, qid))

    writer.flush()
    writer.close()


def load_squad(file="../data/prepared_squad_dev.txt"):
    data = codecs.open(file, 'r', "utf-8").read().split("\n\n")

    x = [[], []]
    y = []
    ids = []
    for content in data:
        if content == "":
            continue
        lines = content.split("\n")
        x[0].append([int(j) for j in lines[0].split()])
        x[1].append([int(j) for j in lines[1].split()])
        y.append([int(i) for i in lines[2].split()])
        ids.append(lines[3])

    return [np.array(x[0]), np.array(x[1])], [np.array(y)[:, 0], np.array(y)[:, 1]], ids
