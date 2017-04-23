import json
import re
import codecs
from nltk.corpus import stopwords
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from nltk.parse.stanford import StanfordDependencyParser
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import numpy as np
import os
from collections import OrderedDict

pos_jar = '../stanford-postagger-2016-10-31/stanford-postagger-3.7.0.jar'
pos_model = '../stanford-postagger-2016-10-31/models/english-bidirectional-distsim.tagger'
pos_tagger = StanfordPOSTagger(pos_model, pos_jar, java_options='-mx10g')
pos_tags = ["", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
        "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP",
        "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

ner_jar = "../stanford-ner-2016-10-31/stanford-ner.jar"
ner_model = "../stanford-ner-2016-10-31/classifiers/english.muc.7class.distsim.crf.ser.gz"
ner_tagger = StanfordNERTagger(ner_model, ner_jar, java_options='-mx10g')

parser_jar = "../stanford-parser-full-2016-10-31/stanford-parser.jar"
parser_model_jar = "../stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar"
parser_model_path = "../stanford-parser-full-2016-10-31/englishPCFG.ser.gz"
parser = StanfordDependencyParser(parser_jar, parser_model_jar, parser_model_path, java_options='-mx100g')

wnl = WordNetLemmatizer()
porter = PorterStemmer()


def create_feature(query, output):
    # pos_list = pos_tagger.tag(query)
    ner_list = ner_tagger.tag(query)
    # print "pos list", pos_list
    dep_str = next(parser.parse(query)).to_conll(4)[:-1]
    dep_list = [line.split('\t') for line in dep_str.split('\n')]

    i = 0
    j = 0
    seq = []
    while i < len(query):
        if j < len(dep_list) and query[i] == dep_list[j][0]:
            seq.append((query[i], ner_list[i][1],
                        dep_list[j][1], dep_list[j][2], dep_list[j][3]))
            j += 1
        else:
            seq.append((query[i], ner_list[i][1], query[i], "_", "_"))
        i += 1

    i = 0
    while i < len(query):
        word = seq[i][0]
        # pos = seq[i][1]
        ner = seq[i][2]
        dependency = seq[i][2:]

        output.write("%s\t" % word)
        output.write("%d\t" % i)
        output.write("%s\t" % wnl.lemmatize(word))
        output.write("%s\t" % porter.stem(word))
        # output.write("%s\t" % pos)
        output.write("%s\t" % ner)
        output.write("%s\t%s\t%s\t" % (dependency[0], dependency[1], dependency[2]))
        output.write("1\t" if word in stopwords.words() else "0\t")
        # output.write(model[word])
        output.write("\n")
        i += 1
    output.write("\n")


def load_data(path):
    # m = 0  # temp!!!!!!!!
    for line in codecs.open(path, "r", "utf-8").readlines():
        data = json.loads(line)
        answer = data["answers"]
        if len(answer) != 1:
            continue
        query_id = data["query_id"]
        query = word_tokenize(data["query"])
        answers = word_tokenize(answer[0])
        passages = data["passages"]

        query_url = "../feature-output-files/queries/" + str(query_id) + ".txt"
        answer_url = "../feature-output-files/answers/" + str(query_id) + ".txt"
        passage_url = "../feature-output-files/passages/" + str(query_id) + ".txt"

        query_file = codecs.open(query_url, "w", "utf-8")
        answer_file = codecs.open(answer_url, "w", "utf-8")
        passage_file = codecs.open(passage_url, "w", "utf-8")

        try:
            create_feature(query, query_file)
            create_feature(answers, answer_file)

            for passage in passages:
                # is_selected = passage["is_selected"]
                passage_text = passage["passage_text"]
                sentences = sent_tokenize(passage_text)
                for sentence in sentences:
                    # print "!!!!!!!!!!!!!!!!!!", sentence   # temp !!!!!!
                    s = word_tokenize(sentence)
                    create_feature(s, passage_file)
                passage_file.write("\n")
            passage_file.write("\n")
        except:
            query_file.flush()
            answer_file.flush()
            passage_file.flush()

            query_file.close()
            answer_file.close()
            passage_file.close()

            os.remove(query_url)
            os.remove(answer_url)
            os.remove(passage_url)
            # error_list = np.append(error_list, query_id)
            print "error ID:", query_id
            continue

        query_file.flush()
        answer_file.flush()
        passage_file.flush()

        query_file.close()
        answer_file.close()
        passage_file.close()

        # if m == 2:  # temp !!!!!!
        #     exit()   # temp !!!!!!
        # m += 1   # temp !!!!!!
        print query_id


def parse_data(path):
    token_freq_dict = {}
    for line in codecs.open(path, "r", "utf-8").readlines():
        data = json.loads(line)

        query_tokens = word_tokenize(data["query"].lower())
        for query_token in query_tokens:
            if query_token not in token_freq_dict:
                token_freq_dict[query_token] = 0
            token_freq_dict[query_token] += 1

        passages = data["passages"]
        for passage in passages:
            passage_text = passage["passage_text"]
            sentences = sent_tokenize(passage_text.lower())
            for sentence in sentences:
                passage_tokens = word_tokenize(sentence)
                for passage_token in passage_tokens:
                    if passage_token not in token_freq_dict:
                        token_freq_dict[passage_token] = 0
                    token_freq_dict[passage_token] += 1

        answers = data["answers"]
        for answer in answers:
            answer_tokens = word_tokenize(answer.lower())
            for answer_token in answer_tokens:
                if answer_token not in token_freq_dict:
                    token_freq_dict[answer_token] = 0
                token_freq_dict[answer_token] += 1

    sorted_token_freq = OrderedDict(sorted(token_freq_dict.items(), key=lambda x: x[1], reverse=True))
    token_to_index = {t:i+1 for i, t in enumerate(sorted_token_freq.keys())}
    index_to_token = {i+1:t for i, t in enumerate(sorted_token_freq.keys())}
    with open("../feature-token-index/token_feature.json", "w") as output:
        json.dump([token_to_index, index_to_token], output, indent=4)
        output.flush()

# ===============================================================================================
# error_id_list = np.array([])
# model = gensim.models.Word2Vec.load("../wiki_sg/word2vec.bin")
# load_data("../dev_v1.1.json/dev_v1.1.json")
# load_data("../data.txt")
parse_data("../input_data/dev_v1.1.json")
