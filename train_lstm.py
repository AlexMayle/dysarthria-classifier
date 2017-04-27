import time
import tensorflow as tf
import math
import pickle
import numpy as np

from sklearn import metrics
from python_speech_features import mfcc
import preprocess

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

    # Normalize to 0-mean, unit-variance
    train[0] = preprocess.meanAndVarNormalize(train[0])
    test[0] = preprocess.meanAndVarNormalize(test[0])

    # It requires too much memory (roughfully 32GB)
    # to hold the covariance matrix neccessary to do
    # ZCA whitening, so we'll skip for now
    #train[0] = preprocess.zcaWhiten(train[0])
    #test[0] = preprocess.zcaWhiten(test[0])

    return train[0], train[1], test[0], test[1]

  def getExampleLengths(self, sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    lengths = tf.cast(lengths, tf.int32)
    return lengths

  def getLastOutput(self, output, exampleLengths):
    shape = tf.shape(output)
    batch_size = shape[0]
    max_length = shape[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (exampleLengths - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

  def test(self, sess, pred_op, X, data, Y, labels, batch_size, is_train):
    """
      sess:        Session to run pred_op in
      pred_op:     Op to get predictions from model
      X:           Placeholder to feed data into
      data:        data to be feed into X placeholder
      Y:           Placeholder to feed labels into
      labels:      labels to be placed into Y
      batch_size:  Batch size to use when interating through data
      is_train:    Placeholder to designate if its training or testing time

      Returns:
        Accuracy, precision, and recall of the model as measured by
        the models predictions and ground truth labels
    """
    
    numExamples = len(data)
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    predictions = np.zeros([numExamples])
    s = 0
    while s < numExamples:
      e = min(s + batch_size, numExamples)
      batch_x = data[s : e]
      batch_x, _ = preprocess.padBatch(batch_x)
      batch_y = labels[s : e]
      predictions[s:e] = sess.run(pred_op, feed_dict={X: batch_x, Y: batch_y, is_train: False})
      s = e

    accuracy = np.sum(predictions == labels) / numExamples
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)

    return accuracy, precision, recall
  
  def model_1(self, X, hidden_size, is_train):
    # ======================================================================
    # Single Layer LSTM over full example sequence
    # Expected: ~87.4%

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

  def model_2(self, X, hidden_size):
    # ======================================================================
    # Two-Layer LSTM over full example sequence
    # Expected: ~88.0%

    # Calculate the actual length of each example
    # so the RNN does not work on the padded sections
    # of each example
    lengths = self.getExampleLengths(X)
    with tf.name_scope("lstm"):
      lstmCell = tf.contrib.rnn.LSTMBlockCell(hidden_size)
      stackedLstmCell = tf.contrib.rnn.MultiRNNCell([lstmCell] * 2)
      output, state = tf.nn.dynamic_rnn(stackedLstmCell, X,
                                        sequence_length= lengths,
                                        dtype= tf.float32)
    # Get the last output for each example
    lastOutputs = self.getLastOutput(output, lengths)
    return lastOutputs


  def model_3(self, X, hidden_size):
    # ======================================================================
    # Two-Layer LSTM over full example sequence
    # Expected: ~88.0%

    # Calculate the actual length of each example
    # so the RNN does not work on the padded sections
    # of each example
    lengths = self.getExampleLengths(X)
    with tf.name_scope("lstm"):
      lstmCell = tf.contrib.rnn.LSTMBlockCell(hidden_size)
      output, _ = tf.nn.bidirectional_dynamic_rnn(lstmCell, lstmCell, X,
                                                      sequence_length= lengths,
                                                      dtype= tf.float32)
    # Get the last output for each example, from the front-pass
    # and the backwards-pass
    lastForwardOutputs = self.getLastOutput(output[0], lengths)
    lastBackwardOutputs = self.getLastOutput(output[1], lengths)
    lastOutputs = tf.concat([lastForwardOutputs, lastBackwardOutputs], axis= 1)
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

      # Choose RNN Model
      if self.mode == 1: # 1-Layer LSTM
        features = self.model_1(X, hidden_size, is_train)
      if self.mode == 2: # 2-Layer LSTM
        features = self.model_2(X, hidden_size)
      if self.mode == 3: # 1-Layer Bidirectional LSTM
        features = self.model_3(X, hidden_size)
        # We need to double the hidden_size now, as we
        # are concatenating the lstm output from the forward
        # and backward pass and sending it to our softmax
        # layer
        hidden_size = hidden_size * 2
        
      # Define softmax layer, use the features.
      softmax_W1 = tf.Variable(tf.random_uniform([hidden_size, class_num]),
                               name= "softmax-weights")
      softmax_b1 = tf.Variable(tf.zeros([class_num]),
                               name= "softmax-bias")
      logits = tf.matmul(features, softmax_W1) + softmax_b1

      # Define loss function, use the logits.
      weights = [param for param in tf.trainable_variables() if "weights" in param.name]
      regulizer = tf.contrib.layers.l2_regularizer(decay)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=Y))
      loss = loss + tf.contrib.layers.apply_regularization(regulizer, weights)
      
      # Define training op, use the loss.
      train_op = tf.train.AdamOptimizer().minimize(loss)

      # Define accuracy op.
      pred = tf.cast(tf.argmax(logits, axis= 1), "int32")
      accuracy = tf.reduce_sum(tf.cast(tf.equal(pred, Y), "float"))

      has_GPU = True
      if has_GPU:
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_option)
      else:
        config = tf.ConfigProto()

      # Create TensorFlow session with GPU setting.
      with tf.Session(config=config).as_default() as sess:
        tf.global_variables_initializer().run()

        # Override the high-level LSTM model's
        # default hidden state initialization
        lstmWeights = [p for p in tf.trainable_variables() if "lstm_cell/weights" in p.name]
        lstm_init = tf.Variable(tf.truncated_normal([hidden_size + 13, hidden_size * 4],
                                                    stddev=0.75))
        lstmWeights[0].assign(lstm_init)
        
        # Train Loop
        for i in range(num_epochs):
          print(20 * '*', 'epoch', i+1, 20 * '*')
          start_time = time.time()
          s = 0
          while s < len(trainX):
            e = min(s + batch_size, len(trainX))
            batch_x = trainX[s : e]
            batch_x, outputLength = preprocess.padBatch(batch_x)
            batch_y = trainY[s : e]
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
            s = e          
          end_time = time.time()
          print ('the training took: %d(s)' % (end_time - start_time))

          # Test the model incrementally, only care about accureacy during each epoch
          accuracy, _, _ = self.test(sess, pred, X, testX, Y, testY, batch_size, is_train)
          print ('accuracy of the trained model %f' % accuracy)

        # After all epochs, calculate accuracy, precision, and recall
        accuracy, precision, recall = self.test(sess, pred, X, testX, Y, testY,
                                                batch_size, is_train)
        return accuracy, precision, recall
