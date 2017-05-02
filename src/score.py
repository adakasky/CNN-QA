from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
from lstm_char_model import LSTMCharModel
from utils import load_squad, get_prediction
from rouge import Rouge


def load_data(references, hypothesis):
    hypo = {idx: [lines.strip()] for (idx, lines) in enumerate(hypothesis)}
    raw_refs = [map(str.strip, r) for r in zip(references)]
    refs = {idx: rr for idx, rr in enumerate(raw_refs)}
    return refs, hypo

def score(ref, hypo):
    scorer = Rouge()
    method = "ROUGE_L"

    final_scores = {}

    score, scores = scorer.compute_score(ref, hypo)
    if type(score) == list:
        for m, s in zip(score, score):
            final_scores[m] = s
    else:
        final_scores[method] = score
    return final_scores

if __name__ == '__main__':

    ROUGE_L = 0.
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

    reference, hypothesis = get_prediction(lstm, x_test, y_test)

    for ref in reference:
        name = reference.index(ref)
        f = codecs.open('../reference/ref' + str(name), 'w', 'utf-8')
        f.write(ref)
        f.close()

    for hyp in hypothesis:
        name = hypothesis.index(hyp)
        f = codecs.open('../hypothesis/hyp'+ str(name),'w', 'utf-8')
        f.write(hyp)
        f.close()

    '''
    for ref, hyp in zip(reference, hypothesis):

        r, p = load_data(ref, hyp)
        score_map = score(r, p)
        ROUGE_L += score_map['ROUGE_L']
    print('Average Metric Score for All Review Summary Pairs:')
    print('Rouge:', ROUGE_L/len(reference))
    '''
