from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rouge import Rouge
import glob
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

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

    ROUGE_L = 0.0

    hyp_file = glob.glob('../hypothesis/*')
    ref_file = glob.glob('../reference/*')
    count = []

    for reference_file, hypothesis_file in zip(ref_file, hyp_file):
        with open(reference_file) as rf:
            reference = rf.readlines()

        with open(hypothesis_file) as hf:
            hypothesis = hf.readlines()

        ref, hypo = load_data(reference, hypothesis)
        score_map = score(ref, hypo)
        count.append(score_map['ROUGE_L'])
        ROUGE_L += score_map['ROUGE_L']


    print('Average Metric Score for All Review Summary Pairs:')
    print('Rouge:', ROUGE_L/min(len(hyp_file) , len(ref_file)) )
