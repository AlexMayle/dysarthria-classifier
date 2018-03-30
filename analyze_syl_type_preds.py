from operator import itemgetter
from functools import reduce
from heapq import nlargest
import pickle

import argparse
import numpy as np
import sklearn.metrics as sklm
import matplotlib.pyplot as plt

from preprocess import splitIntoPatches, splitDataAndLabels

parser = argparse.ArgumentParser('Analyze Experiment Results')
parser.add_argument('--z_path', "-z",
                    type=str,
                    required=True,
                    default="",
                    help='Location to save the resulting graphs')
parser.add_argument('--zz_path', '-zz',
                    type=str,
                    default="",
                    required=True,
                    help='Location of results to analyze')
parser.add_argument('--norm_path', '-n',
                    type=str,
                    default="",
                    required=True,
                    help='Location of results to analyze')
parser.add_argument('--save_path', '-s',
                    type=str,
                    default="",
                    required=True,
                    help='Location of results to analyze')




FLAGS = None
FLAGS, unparsed = parser.parse_known_args()

PREDICTION_PATH = "out/predictions/"
def syllable_frequencies(syllable_annotations):
    return {k: syllable_annotations.count(k) for k in syllable_annotations}

def patient_variances(predictions, annotations):
    annotations = [[name] * len(syl_list) for name, _, syl_list, _ in annotations]
    annotations = reduce(lambda x, y: x + y, annotations)
    print(len(predictions), len(predictions[0]), len(annotations))
    assert len(predictions[0]) == len(annotations)
    variances = syllable_variances(predictions, annotations)
    return variances

def syllable_variances(predictions, syllable_annotations):
    counts = syllable_frequencies(syllable_annotations)
    syllable_var_dicts = []
    for pred_set in predictions:
        sums = dict()
        sums = {k: sums.get(k, 0) + v for k, v in zip(syllable_annotations, pred_set)}
        avgs = {k: v / counts[k] for (k, v) in sums.items()}
        residuals = [(k, (v - avgs[k]) ** 2) for k, v in zip(syllable_annotations, pred_set)]
        res_sums = dict()
        res_sums = {k: res_sums.get(k, 0) + v for (k, v) in residuals}
        variances = {k: v / counts[k] for (k, v) in res_sums.items() if counts[k] > 10}
        syllable_var_dicts.append(variances)

    return syllable_var_dicts


def syllable_accuracies(labels, predictions, annotations):
    # ungroup annoations
    syllable_acc_dicts = []
    for pred_set in predictions:
        assert len(pred_set) == len(labels) == len(annotations)
        annotated_preds = zip(pred_set, labels, annotations)
        annotated_preds = sorted(annotated_preds, key=lambda x: x[0])
        i = 1
        while i < len(annotated_preds):
            if annotated_preds[i - 1][1] <= annotated_preds[i][1]:
                i += 1
            else:
                break
        counts = syllable_frequencies(annotations)
        misses = dict()
        while i < len(annotated_preds):
            if annotated_preds[i][1] == 0:
                misses[annotated_preds[i][2]] = misses.get(annotated_preds[i][2], 0) + 1
            i += 1
            syllable_accs = {k: v / counts[k] for k, v in misses.items() if counts[k] > 10}
        syllable_acc_dicts.append(syllable_accs)
    return syllable_acc_dicts

def pr_and_roc_curves(labels, predictions):

    def pr_and_roc_curve(labels, predictions):
        pr_curve_y, pr_curve_x, th_1 = sklm.precision_recall_curve(labels, predictions)
        roc_curve_x, roc_curve_y, th_2 = sklm.roc_curve(labels, predictions,
                                                        drop_intermediate=False)
        print('pr thresholds', th_1)
        print('roc threshholds', th_2)
        pr_curve = (pr_curve_x, pr_curve_y)
        roc_curve = (roc_curve_x, roc_curve_y)
        return pr_curve, roc_curve


    curve_data = []
    for pred_set in predictions:
        curve_data.append(pr_and_roc_curve(labels, pred_set))
    pr_curves, roc_curves = zip(*curve_data)
    return pr_curves, roc_curves


def plot_roc_and_pr_curves(title, curve_data):

    def plot_curve(title, plot_args, type_='pr'):
        lines = plt.plot(*plot_args)
        plt.title(title)
        plt.axis([0, 1, 0, 1])
        plt.legend(lines, ['z-LSTM-1', 'z-LSTM-2', 'z-Bi-LSTM-1', 'zz-LSTM-1', 'zz-LSTM-2', 'zz-Bi-LSTM-1', 'LSTM-1', 'LSTM-2', 'Bi-LSTM-1'])
        if type_ == 'pr':
            plt.xlabel('Recall')
            plt.ylabel('Precision')
        else:
            plt.plot([0, 1], [0, 1], 'r-', linestyle='dashed', lw=1)
            plt.xlabel('FPR')
            plt.ylabel('TPR')
        if FLAGS.save_path:
            plt.savefig("%s/%s-%s.png" % (FLAGS.save_path, title, type_),
                        bbox_inches="tight")
        plt.show()

    def format_curves(curves):
        format_strs = ['r.-', 'ro-', 'r^-', 'b.-', 'bo-', 'b^-', 'g.-', 'go-', 'g^-']
        args = []
        for curve, format_ in zip(curves, format_strs):
            x, y = curve
            args += [x, y, format_]
        return args
    pr_curves, roc_curves = zip(*curve_data)
    pr_curves = [subitem for item in pr_curves for subitem in item]
    roc_curves = [subitem for item in roc_curves for subitem in item]
    pr_args = format_curves(pr_curves)
    roc_args = format_curves(roc_curves)
    plot_curve(title, pr_args, type_='pr')
    plot_curve(title, roc_args, type_='roc')

def auc_scores(labels, predictions):
    scores = []
    for pred_set in predictions:
        scores.append(sklm.roc_auc_score(labels, pred_set, average=None))
    return scores

def convert_syllables_to_noisy_or(syllable_predictions, syllables_per_patient):
    def seperate_by_speaker(prediction_set, syllable_per_pat):
        grouped_syllables = []
        i = 0
        for j in syllable_per_pat:
            grouped_syllables.append(prediction_set[i:i+j])
            i += j
        return grouped_syllables


    fo_predictions = []
    for pred_set in syllable_predictions:
        negated_probs = 1 - pred_set + 10**-13
        log_probs = np.log(negated_probs)
        speaker_log_probs = seperate_by_speaker(pred_set, syllables_per_patient)
        fo_scores = [np.mean(x) for x in speaker_log_probs]
        fo_predictions.append(np.asarray(fo_scores))
    return fo_predictions

if __name__ == '__main__':
    curve_data = []
    for path in (FLAGS.z_path, FLAGS.zz_path, FLAGS.norm_path):
        with open("%s/lstm-1/results" % path, "rb") as f:
            lstm_1_preds = pickle.load(f)
        with open("%s/lstm-2/results" % path, "rb") as f:
            lstm_2_preds = pickle.load(f)
        with open("%s/bi-lstm-1/results" % path, "rb") as f:
            bi_lstm_1_preds = pickle.load(f)

        lstm_1_preds, labels = lstm_1_preds
        lstm_2_preds, _ = lstm_2_preds
        bi_lstm_1_preds, _ = bi_lstm_1_preds
        assert len(lstm_1_preds) == len(lstm_2_preds) == len(bi_lstm_1_preds)
        _, sm_preds, _ = zip(lstm_1_preds, lstm_2_preds, bi_lstm_1_preds)
        sm_pr_curves, sm_roc_curves = pr_and_roc_curves(labels, sm_preds)
        curve_data.append((sm_pr_curves, sm_roc_curves))

    # plot the curves
    plot_roc_and_pr_curves("Soft Majority Evaluation", curve_data)

    # calculate auc 
    syl_auc_scores = auc_scores(syl_labels, syl_preds)
    sm_auc_scores = auc_scores(labels, sm_preds)
    fo_auc_scores = auc_scores(labels, fo_preds)
    # print auc
    lstm_1_auc, lstm_2_auc, bi_lstm_1_auc = zip(syl_auc_scores, sm_auc_scores,
                                                fo_auc_scores)
    print("=" * 10, " AUC ", "=" * 10)
    print(" " * 8, "Syllable, Soft Majority, Fuzzy Or")
    print("LSTM-1 %s, %s, %s" % lstm_1_auc)
    print("LSTM-2 %s, %s, %s" % lstm_2_auc)
    print("Bi-LSTM-1 %s, %s, %s" % bi_lstm_1_auc)
