"""
dilated CNN-CNN Encoder-Decoder model of question answering

@author: Ao Liu, Zhuodong Huang, Zitao Wang
"""

from __future__ import division
from __future__ import print_function
import json
import gzip

from gensim.models import Word2Vec

embeddings = Word2Vec.load("../data/word2vec.bin")


def load_data(f):
    lines = gzip.open(f, 'r').readlines()

    for line in lines:
        content = json.loads(line.decode("utf-8"))
        if len(content["answers"]) > 1:
            continue
        query = content["query"]
        passages = content["passages"]
        indicator = map(lambda p: p["is_selected"], passages)
        passages = map(lambda p: p["passage_text"], passages)
        answers = content["answers"]

        exit()

