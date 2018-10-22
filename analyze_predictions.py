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
parser.add_argument('--pred_path', '-p',
                    type=str,
                    default="",
                    required=True,
                    help=('Directory to find the resulting predictions from the \
                          experiment. Also saves graphs here if --save_path is \
                          not provided.'))
parser.add_argument('--data_path', '-d',
                    type=str,
                    default="",
                    required=True,
                    help=('Directory of pickled object containing the dataset \
                           used for the experiment'))
parser.add_argument('--save_path', "-s",
                    type=str,
                    default="",
                    help=('Directory to save the resulting graphs. \
                           Defaults to --pred_path.'))


FLAGS = None
FLAGS, unparsed = parser.parse_known_args()


def pr_and_roc_curves(labels, predictions):

    def pr_and_roc_curve(labels, predictions):
        pr_curve_y, pr_curve_x, th_1 = sklm.precision_recall_curve(labels, predictions)
        roc_curve_x, roc_curve_y, th_2 = sklm.roc_curve(labels, predictions,
                                                        drop_intermediate=False)

        pr_curve = (pr_curve_x, pr_curve_y)
        roc_curve = (roc_curve_x, roc_curve_y)
        return pr_curve, roc_curve


    curve_data = []
    for pred_set in predictions:
        curve_data.append(pr_and_roc_curve(labels, pred_set))
    pr_curves, roc_curves = zip(*curve_data)
    return pr_curves, roc_curves


def plot_roc_and_pr_curves(title, pr_curves, roc_curves):

    def plot_curve(title, plot_args, type_='pr'):
        lines = plt.plot(*plot_args)
        plt.title(title)
        plt.axis([0, 1, 0, 1])
        plt.legend(lines, ['LSTM-1', 'LSTM-2', 'Bi-LSTM-1'])
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
        if len(curves[0][0]) > 100:
            format_strs = ['-'] * 3
        else:
            format_strs = ['.-', 'o-', '^-']
        args = []
        for curve, format_ in zip(curves, format_strs):
            x, y = curve
            args += [x, y, format_]
        return args

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
    FLAGS.save_path = FLAGS.pred_path if FLAGS.save_path == '' else FLAGS.save_path
    with open("%s/lstm-1/results" % FLAGS.pred_path, "rb") as f:
        lstm_1_preds = pickle.load(f)
    with open("%s/lstm-2/results" % FLAGS.pred_path, "rb") as f:
        lstm_2_preds = pickle.load(f)
    with open("%s/bi-lstm-1/results" % FLAGS.pred_path, "rb") as f:
        bi_lstm_1_preds = pickle.load(f)
    with open("%s.pkl" % FLAGS.data_path, "rb") as f:
        full_data_set = pickle.load(f)
    with open("%s_meta.pkl" % FLAGS.data_path, "rb") as f:
        full_annotations = pickle.load(f)
        full_annotations = full_annotations[9:]

    # Get labels
    syl_labels = splitIntoPatches(full_data_set[9:])
    syl_labels = splitDataAndLabels(syl_labels)[1]
    # Get predictions
    lstm_1_preds, labels = lstm_1_preds
    lstm_2_preds, _ = lstm_2_preds
    bi_lstm_1_preds, _ = bi_lstm_1_preds
    assert len(lstm_1_preds) == len(lstm_2_preds) == len(bi_lstm_1_preds)
    syl_preds, sm_preds, fo_preds = zip(lstm_1_preds, lstm_2_preds, bi_lstm_1_preds)
    syl_per_patient = [len(x[0]) for x in full_data_set[9:]]
    syl_preds = [np.concatenate(x) for x in syl_preds]
    fo_preds = convert_syllables_to_noisy_or(syl_preds, syl_per_patient)
    assert len(syl_labels) == len(syl_preds[0])
    # Get annotations
    syl_annotations = [annotation[2] for annotation in full_annotations]
    syl_annotations = reduce(lambda x, y: x + y, syl_annotations)
    pat_annotations = [[name] * len(syl_list) for name, _, syl_list, _, in full_annotations]
    pat_annotations = reduce(lambda x, y: x + y, pat_annotations)

    # Calculate pr and roc curve data
    syl_pr_curves, syl_roc_curves = pr_and_roc_curves(syl_labels, syl_preds)
    sm_pr_curves, sm_roc_curves = pr_and_roc_curves(labels, sm_preds)
    fo_pr_curves, fo_roc_curves = pr_and_roc_curves(labels, fo_preds)

    # plot the curves
    plot_roc_and_pr_curves("Syllable Evaluation", syl_pr_curves, syl_roc_curves)
    plot_roc_and_pr_curves("Soft Majority Evaluation", sm_pr_curves, sm_roc_curves)
    plot_roc_and_pr_curves("Noisy Or Evaluation (log space)", fo_pr_curves, fo_roc_curves)

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
