from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import pickle

import tensorflow as tf

from train_baseline import Baseline
from train_lstm import LSTMNet
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
parser.add_argument('--input_data_dir', 
                    type=str, 
                    default='../mnist/data', 
                    help='Directory to put the training data.')
parser.add_argument('--expanded_data', 
                    type=str, 
                    default='../mnist/data/mnist_expanded.pkl.gz', 
                    help='Directory to put the extended mnist data.')
parser.add_argument('--log_dir', 
                    type=str, 
                    default='logs', 
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
                    default=str(44100),
                    help="Sample rate of wav data")

FLAGS = None
FLAGS, unparsed = parser.parse_known_args()
mode = int(sys.argv[1])

# ======================================================================
#  STEP 1: Train a baseline model.

if mode == 1:
    FLAGS.batch_size = 30
    nn = Baseline(1)
    accuracy = nn.train_and_evaluate(FLAGS)

    # Output accuracy.
    print(20 * '*' + ' model 1 (Baseline)' + 20 * '*')
    print('accuracy is %f' % accuracy)
    print()
    exit()

# ====================================================================
# STEP 2: Train an LSTM model.

old_lstm = LSTMNet(mode=mode-1)
lstm = LstmNet(mode=mode-1)
"""trn_data, trn_targets, val_data, val_targets, tst_data, tst_targets = old_lstm.read_data()
sess, path = lstm.train(trn_data, trn_targets,
                        val_data=val_data,
                        val_targets=val_targets,
                        test_data=tst_data,
                        test_targets=tst_targets,
                        num_epochs=1)
sess.close()

sess = lstm.load_pretrained_weights()
grp_train_data, grp_test_data = old_lstm.read_grouped_data()
with sess:
    stats, sess = lstm.patient_level_evaluate(grp_train_data, grp_test_data)
accuracy, precision, recall = stats
print(20 * '*' + ' model 2 (LSTM 1) ' + 20 * '*')
print('accuracy is %f' % accuracy)
print("precision is %f" % precision)
print("recall is %f" % recall)
print()
"""
with open('mfcc_full_set.pkl', 'rb') as f:
    full_data_set = pickle.load(f)
cross_validation_data = full_data_set[9:]
validation_data = full_data_set[:9]
lstm.k_fold_cross_validation(cross_validation_data, validation_data)
