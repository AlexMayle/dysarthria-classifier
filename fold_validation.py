from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pickle
import copy

from lstm_model import LstmNet

# Set parameters for Sparse Autoencoder
parser = argparse.ArgumentParser('CNN Exercise.')
parser.add_argument('--learning_rate', 
                    type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=40,
                    help='Number of epochs to run trainer.')
parser.add_argument('--decay',
                    type=float,
                    default=0.008,
                    help='Decay rate of l2 regularization.')
parser.add_argument('--restore',
                    type=int,
                    default=0,
                    help='Set to 1 to restore weights from previous session')
parser.add_argument('--batch_size', 
                    type=int,
                    default=64, 
                    help='Batch size. Must divide evenly into the dataset sizes.')
parser.add_argument('--input_path', 
                    type=str, 
                    default='', 
                    help='Directory to put the training data.')
parser.add_argument('--expanded_data', 
                    type=str, 
                    default='../mnist/data/mnist_expanded.pkl.gz', 
                    help='Directory to put the extended mnist data.')
parser.add_argument('--output_path',
                    type=str,
                    required=True,
                    help='Directory to put logging.')
parser.add_argument('--visibleSize',
                    type=int,
                    default=str(28 * 28),
                    help='Used for gradient checking.')
parser.add_argument('--hiddenSize',
                    type=int,
                    default='200',
                    help='.')
parser.add_argument("--sample_rate",
                    type=int,
                    default="44100",
                    help="Sample rate of wav data")
parser.add_argument("--numceps",
                    type=int,
                    default="13",
                    help="Number of cepstrums per MFCC feature")
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()
mode = int(sys.argv[1])


def calculate_roc_and_pr_curve(predictions, labels):
    labels = copy.deepcopy(labels)
    # Sort labels by positive probability
    dataset = list(zip(predictions, labels))
    dataset.sort(key=lambda x: x[0])
    _, labels = zip(*dataset)

    #  false_neg[i] is the number of FN given threshhold prediction[i]
    #  false_neg[-1] corresponds to threshhold positive infinity
    false_neg = [0]
    num_examples = len(predictions)
    for i in range(1, num_examples + 1):
        false_neg.append(false_neg[i - 1] + labels[i - 1])

    true_pos = []
    false_pos = []
    for i in range(num_examples + 1):
        true_pos.append(false_neg[-1] - false_neg[i])
        false_pos.append(num_examples - i - true_pos[i])

    prec = []
    recall = []
    for i in range(num_examples):
        prec.append(true_pos[i] / (num_examples - i))
        recall.append(true_pos[i] / false_neg[-1])
    prec.append(1)
    recall.append(0)

    # Scale
    positive_examples = true_pos[0]
    negative_examples = num_examples - positive_examples
    true_pos = list(map(lambda x: x / positive_examples, true_pos))
    false_pos = list(map(lambda x: x / negative_examples, false_pos))

    return false_pos, true_pos, recall, prec

print("input: %s\n output: %s" % (FLAGS.input_path, FLAGS.output_path))

lstm = LstmNet(mode=mode-1, input_size=FLAGS.numceps)
with open(FLAGS.input_path, 'rb') as f:
    full_data_set = pickle.load(f)
cross_validation_data = full_data_set[9:]
validation_data = full_data_set[:9]
predictions = lstm.k_fold_cross_validation(cross_validation_data, validation_data,
                                           num_epochs=FLAGS.num_epochs,
                                           output_path=FLAGS.output_path)
syl_preds = predictions[0]
sm_preds = predictions[1]
fo_preds = predictions[2]
_, labels = zip(*cross_validation_data)

# save to disk
with open(FLAGS.output_path + "/results", "wb") as f:
    pickle.dump((predictions, labels), f)
