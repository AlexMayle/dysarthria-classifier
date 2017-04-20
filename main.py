from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import gzip
import pickle

import numpy
from python_speech_features import mfcc
import tensorflow as tf

from loadData import loadDataSet
from train_cnn import ConvNet
from train_lstm import LSTMNet

# Set parameters for Sparse Autoencoder
parser = argparse.ArgumentParser('CNN Exercise.')
parser.add_argument('--learning_rate', 
                    type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=60, 
                    help='Number of epochs to run trainer.')
parser.add_argument('--decay',
                    type=float,
                    default=0.1, 
                    help='Decay rate of l2 regularization.')
parser.add_argument('--batch_size', 
                    type=int,
                    default=20, 
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
                    default='100',
                    help='.')
parser.add_argument("--sample_rate",
                    type=int,
                    default=str(44100),
                    help="Sample rate of wav data")
 
FLAGS = None
FLAGS, unparsed = parser.parse_known_args()
mode = int(sys.argv[1])

# ======================================================================
#  STEP 0: Load data
#trainSet = loadDataSet("train_set.pkl")
#testSet = loadDataSet("test_set.pkl")

# ======================================================================
#  STEP 1: Train a baseline model.

if mode == 1:
  FLAGS.hiddenSize = 200
  FLAGS.batch_size = 30
  FLAGS.num_epochs = 100
  cnn = ConvNet(1)
  accuracy = cnn.train_and_evaluate(FLAGS)

  # Output accuracy.
  print(20 * '*' + ' model 1 (Baseline)' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()


# ====================================================================
# STEP 2: Train an LSTM model.

if mode == 2:
  lstm = LSTMNet(1)
  accuracy = lstm.train_and_evaluate(FLAGS, trainSet, testSet)

  # Output accuracy.
  print(20 * '*' + ' model 2 (LSTM 1) ' + 20 * '*')
  print('accuracy is %f' % (accuracy))
  print()
