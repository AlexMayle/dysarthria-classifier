import time
import tensorflow as tf
import math
import pickle
import numpy as np

from python_speech_features import mfcc
from preprocess import *

SAMPLE_RATE = 44100
MFCC_SIZE = 13

########### LSTM Network ##############
class LSTMNet(object):
  def __init__(self, mode):
    self.mode = mode
 
  # Read train, valid and test data.
  def read_data(self):
    """
    Reads training and test set from disk in pickle format
    
    Returns: 
      - A list of numpy arrays containing mfcc features training examples
      - A list of integers as labels for the training examples 
      - A list of numpy arrays containing mfcc feature test examples
      - A list of integers as labels for the test examples
    """
  
    print("[*] Loading mfcc training & testing set from disk")
    with open("mfcc_train_set.pkl", "rb") as f:
      train = pickle.load(f)
    with open("mfcc_test_set.pkl", "rb") as f:
      test = pickle.load(f)
    
    return train[0], train[1], test[0], test[1]

  def getExampleLengths(self, sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    lengths = tf.cast(lengths, tf.int32)
    return lengths

  def getLastOutput(self, output, exampleLengths):
    shape = tf.shape(output)
    print(shape)
    batch_size = shape[0]
    max_length = shape[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (exampleLengths - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
  
  # Baseline model.
  def model_1(self, X, hidden_size):
    # ======================================================================
    # Single Layer LSTM over full example sequence
    #
    # ----------------- YOUR CODE HERE ----------------------
    #

    # Calculate the actual length of each example
    # so the RNN does not work on the padded sections
    # of each example
    lengths = self.getExampleLengths(X)
    with tf.name_scope("lstm"):
      lstmCell = tf.contrib.rnn.LSTMBlockCell(hidden_size)
      output, state = tf.nn.dynamic_rnn(lstmCell, X, sequence_length= lengths, dtype= tf.float32)
    # Get the last output for each example
    lastOutputs = self.getLastOutput(output, lengths)
    return lastOutputs

  # Entry point for training and evaluation.
  def train_and_evaluate(self, FLAGS):
    class_num     = 2
    num_epochs    = FLAGS.num_epochs
    batch_size    = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    hidden_size   = FLAGS.hiddenSize
    decay         = FLAGS.decay

    trainX, trainY, testX, testY = self.read_data()
    print("[*] Preprocessing done")
    
    with tf.Graph().as_default():
      # Input data
      X = tf.placeholder(tf.float32, [None, None, 13])
      Y = tf.placeholder(tf.int32, [None])
      is_train = tf.placeholder(tf.bool)

      # model 1: base line
      if self.mode == 1:
        features = self.model_1(X, hidden_size)
        
      # Define softmax layer, use the features.
      softmax_W1 = tf.Variable(tf.random_uniform([hidden_size, class_num]),
                               name= "softmax-weights")
      softmax_b1 = tf.Variable(tf.zeros([class_num]),
                               name= "softmax-bias")
      logits = tf.matmul(features, softmax_W1) + softmax_b1

      # Define loss function, use the logits.
      #params = tf.trainable_variables()
      #l2_reg = sum([tf.nn.l2_loss(param) for param in params if "Bias" not in param.name])
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,                                                                    labels=Y))
      # Define training op, use the loss.
      optimizer = tf.train.AdamOptimizer()
      train_op = optimizer.minimize(loss)

      # Define accuracy op.
      pred = tf.cast(tf.argmax(logits, axis= 1), "int32")
      accuracy = tf.reduce_sum(tf.cast(tf.equal(pred, Y), "float"), name= "acc")

      has_GPU = True
      if has_GPU:
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_option)
      else:
        config = tf.ConfigProto()

      # Create TensorFlow session with GPU setting.
      with tf.Session(config=config).as_default() as sess:
        tf.global_variables_initializer().run()
                
        for i in range(num_epochs):
          print(20 * '*', 'epoch', i+1, 20 * '*')
          start_time = time.time()
          s = 0
          while s < len(trainX):
            e = min(s + batch_size, len(trainX))
            batch_x = trainX[s : e]
            batch_x, outputLength = padBatch(batch_x)
            batch_y = trainY[s : e]
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
            s = e          
          end_time = time.time()
          print ('the training took: %d(s)' % (end_time - start_time))
          s = 0
          totalCorrect = 0
          while s < len(testX):
            e = min(s + batch_size, len(testX))
            batch_x = testX[s : e]
            batch_x, _ = padBatch(batch_x)
            batch_y = testY[s : e]
            correct = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, is_train: False})
            totalCorrect += correct
            s = e

          acc = totalCorrect / len(testX)
          print ('accuracy of the trained model %f' % acc)

        return acc
