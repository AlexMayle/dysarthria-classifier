import os
import time
import pickle
import numpy as np

import tensorflow as tf
from sklearn import metrics
from python_speech_features import mfcc
import preprocess

SAMPLE_RATE = 44100
MFCC_SIZE = 13
CHECKPOINT_FILEPATH = "out/model.ckpt"
GROUPED_TESTSET_PATH = "grouped_test_set.pkl"
MFCC_GRPED_TESTSET_PATH = "mfcc_grouped_test_set.pkl"

########### LSTM Network ##############
class LSTMNet(object):
  def __init__(self, mode):
    self.mode = mode

  def read_grouped_data(self):
    if os.path.isfile(MFCC_GRPED_TESTSET_PATH):
      print("[*] Loading mfcc grouped test set from disk")
      with open(MFCC_GRPED_TESTSET_PATH, "rb") as f:
        testSet = pickle.load(f)
    elif os.path.isfile(GROUPED_TESTSET_PATH):
      print("[*] Converting grouped test set to mfcc's")
      testSet = preprocess.loadDataSet(GROUPED_TESTSET_PATH)
      testSet[0] = list(map(lambda x: list(map(lambda y: mfcc(y, samplerate= SAMPLE_RATE), x)),
                       testSet[0]))
      with open(MFCC_GRPED_TESTSET_PATH, "wb") as f:
        pickle.dump(testSet, f)
    else:
      print("[*] Can't find grouped dataset, use createDataset.py --no_patches to create")
      exit(-1)

    # Normalize
    testSet[0] = preprocess.meanAndVarNormalize(testSet[0])

    return testSet[0], testSet[1]

  # Read train, valid jnd test data.
  def read_data(self):
    """
    Reads training and test set from disk in pickle format
    
    Returns: 
      - A list of numpy arrays containing mfcc features training examples
      - A list of integers as labels for the training examples 
      - A list of numpy arrays containing mfcc feature test examples
      - A list of integers as labels for the test examples
    """
  
    print("[*] Loading mfcc training, validation, and testing set from disk")
    with open("mfcc_train_set.pkl", "rb") as f:
      train = pickle.load(f)
    with open("mfcc_val_set.pkl", "rb") as f:
      val = pickle.load(f)
    with open("mfcc_test_set.pkl", "rb") as f:
      test = pickle.load(f)

    # Normalize to 0-mean, unit-variance
    train[0] = preprocess.meanAndVarNormalize(train[0])
    val[0] = preprocess.meanAndVarNormalize(val[0])
    test[0] = preprocess.meanAndVarNormalize(test[0])

    # It requires too much memory (roughfully 32GB)
    # to hold the covariance matrix neccessary to do
    # ZCA whitening, so we'll skip for now
    #train[0] = preprocess.zcaWhiten(train[0])
    #test[0] = preprocess.zcaWhiten(test[0])

    return train[0], train[1], val[0], val[1], test[0], test[1]

  def f_score(self, precision, recall):
    return 2 * precision * recall / (precision + recall)

  def generalizationLoss(self, maxMetric, currentMetric):
    minErr = 1 - maxMetric
    currentErr = 1 - currentMetric
    return currentErr / minErr - 1

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

  def patientLevelEvaluate(self, session, pred_op, data, labels):
    """
      Evaluates the model per patient, rather than on a per syllable
      level

      session:     Session to run pred_op in
      pred_op:     Op to get predictions from model
      data:        data to be fed into X placeholder. Should be a list of
                   of lists, where each inner list contains numpy arrays
                   representing the patients' syllables where each row is
                   an MFCC of the syllable
      labels:      labels to be placed into Y

      Returns:
        Accuracy, precision, and recall of the model as measured by
        the models predictions and ground truth labels
    """
 
    feed_dict = dict()
    feed_dict[self.is_train] = False
    num_examples = len(data)
    predictions = np.zeros([num_examples])

    print("[*] Starting grouped evaluation")
    for i in range(num_examples):
        batch_x, _ = preprocess.padBatch(data[i])
        feed_dict[self.X_placeholder] = batch_x
        feed_dict[self.Y_placeholder] = [labels[i]]
        predictions[i] = session.run(pred_op, feed_dict=feed_dict)

    accuracy = np.sum(predictions == labels) / num_examples
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)

    return accuracy, precision, recall

  def syllableLevelEvaluate(self, session, pred_op, data, labels):
    """
      Evaluates the model per syllable, rather than on a per patient
      level

      session:     Session to run pred_op in
      pred_op:     Op to get predictions from model
      data:        data to be fed into X placeholder. Should be a list of
                   of numpy arrays where each row is an MFCC of the syllable
      labels:      labels to be placed into Y

      Returns:
        Accuracy, precision, and recall of the model as measured by
        the models predictions and ground truth labels
    """

    feed_dict = {}
    feed_dict[self.is_train] = False
    numExamples = len(data)
    predictions = np.zeros([numExamples])
    batch_size = 32                       # doesn't really matter in testing phase

    s = 0
    while s < numExamples:
      e = min(s + batch_size, numExamples)
      batch_x = data[s : e]
      batch_x, _ = preprocess.padBatch(batch_x)
      batch_y = labels[s : e]
      predictions[s:e] = session.run(pred_op, feed_dict={self.X_placeholder: batch_x,
                                                      self.Y_placeholder: batch_y,
                                                      self.is_train: False})
      s = e

    accuracy = np.sum(predictions == labels) / numExamples
    precision = metrics.precision_score(labels, predictions)
    recall = metrics.recall_score(labels, predictions)

    return accuracy, precision, recall

  def model_1(self, X, hidden_size, is_train):
    # ======================================================================
    # Single Layer LSTM over full example sequence with L2 reg
    # Expected: ~87.6%

    # Calculate the actual length of each example
    # so the RNN does not work on the padded sections
    # of each example
    lengths = self.getExampleLengths(X)
    
    with tf.name_scope("lstm"):
      lstmCell = tf.contrib.rnn.LSTMCell(hidden_size)
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
      dropoutlstmCell = tf.contrib.rnn.DropoutWrapper(lstmCell, input_keep_prob= 0.5,
                                                      output_keep_prob= 0.5)
      stackedLstmCell = tf.contrib.rnn.MultiRNNCell([lstmCell, dropoutlstmCell])
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
    hidden_size   = softmax_hidden_size = FLAGS.hiddenSize
    decay         = FLAGS.decay

    # Input / Preprocess data (some preprocessing was
    # (already done and so generally this will just
    # pull the preprocessed dataset from disk
    trainX, trainY, valX, valY, testX, testY = self.read_data()
    print("[*] Preprocessing done")

    # Construct computation graph
    with tf.Graph().as_default():
      # Placeholders
      self.X_placeholder = tf.placeholder(tf.float32, [None, None, 13])
      self.Y_placeholder = tf.placeholder(tf.int32, [None])
      self.is_train = tf.placeholder(tf.bool)

      # Choose RNN Model
      if self.mode == 1: # 1-Layer LSTM
        features = self.model_1(self.X_placeholder, hidden_size, self.is_train)
      if self.mode == 2: # 2-Layer LSTM
        features = self.model_2(self.X_placeholder, hidden_size)
      if self.mode == 3: # 1-Layer Bidirectional LSTM
        features = self.model_3(self.X_placeholder, hidden_size)
        # We need to double the hidden_size now, as we
        # are concatenating the lstm output from the forward
        # and backward pass and sending it to our softmax
        # layer
        softmax_hidden_size = hidden_size * 2

      # Define softmax layer, use the features.
      softmax_W1 = tf.Variable(tf.random_uniform([softmax_hidden_size, class_num]),
                               name= "softmax-weights")
      softmax_b1 = tf.Variable(tf.zeros([class_num]),
                               name= "softmax-bias")
      logits = tf.matmul(features, softmax_W1) + softmax_b1

      # loss + regularization
      weights = [param for param in tf.trainable_variables() if "weights" in param.name]
      if self.mode == 3:
        decay /= 2
      regulizer = tf.contrib.layers.l2_regularizer(decay)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=self.Y_placeholder)
      loss = tf.reduce_mean(loss)

      # Only apply l2 reg to the single-layer LSTM architectures because we can't do dropout
      if self.mode == 1 or 3:
        loss = loss + tf.contrib.layers.apply_regularization(regulizer, weights)

      # Optimizer
      train_op = tf.train.AdamOptimizer().minimize(loss)

      # Generally I prefer to just fetch the pred op
      # and handle the metrics in numPy, but the accuracy
      # op will also get what you want within TensorFlow
      pred = tf.cast(tf.argmax(logits, axis= 1), "int32")
      # Op for predicting based off all of a speakers' character
      # pronunciations
      predWeightsRaw = tf.reduce_max(logits, reduction_indices= 1)
      predWeights = tf.nn.softmax(predWeightsRaw)
      weightedPreds = tf.multiply(predWeights, tf.cast(pred, 'float'))
      groupPred = tf.round(tf.reduce_sum(weightedPreds))

      # Configure GPU use
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
        # default hidden state initialization (it defaults to 0's)
        lstmWeights = [p for p in tf.trainable_variables() if "lstm_cell/weights" in p.name]
        # TODO: Make the stdDev of the initializer a hyperparameter (I already tuned it, tho)
        lstm_init = tf.Variable(tf.truncated_normal([hidden_size + 13, hidden_size * 4],
                                                    stddev=0.25))
        lstmWeights[0].assign(lstm_init)


        # Get the variables we are actually trying to train (basically everything
        # not involving the adam algorithm, which I'm not even sure why those are
        # trainable, I suppose for hyper-parameter tuning?
        params = tf.trainable_variables()
        params = [p for p in params if
                  ("adam" not in p.name and (("weights" in p.name) or "bias" in p.name))]
        saver = tf.train.Saver(params)

        if FLAGS.restore == 1 or FLAGS.restore == 2:
          print("[*] Restoring weights, skipping training")
          saver.restore(sess, CHECKPOINT_FILEPATH)
          if FLAGS.restore == 2:
            data, labels = self.read_grouped_data()
            accuracy, precision, recall = self.patientLevelEvaluate(sess, groupPred, data, labels)
            return accuracy, precision, recall

        maxMetric = 0
        momentumSteps = 0
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
            sess.run(train_op, feed_dict={self.X_placeholder: batch_x,
                                           self.Y_placeholder: batch_y,
                                           self.is_train: True})
            s = e

          end_time = time.time()
          print ('the training took: %d(s)' % (end_time - start_time))

          if i % 2 == 0:
            accuracy, _, _ = self.syllableLevelEvaluate(sess, pred, testX, testY)
            print(accuracy)

          # After epoch 5, start applying early stop regularization using recall as a metric
          # and a 5 step period to wait if the metric will rebound
          if i >= 5:
            _, precision, recall = self.syllableLevelEvaluate(sess, pred, valX, valY)
            f_score = self.f_score(precision, recall)
            if f_score >= maxMetric:
              maxMetric = f_score
              momentumSteps = 0
              saver.save(sess, CHECKPOINT_FILEPATH)
            else:
              GL = self.generalizationLoss(maxMetric, f_score)
              if GL > .175:
                if momentumSteps > 3:
                  print("Stopping Early. . .")
                  saver.restore(sess, CHECKPOINT_FILEPATH)
                  break
                else:
                  momentumSteps += 1

        print("[*] weights saved at %s" % CHECKPOINT_FILEPATH)
        saver.save(sess, CHECKPOINT_FILEPATH)

        # After all epochs, calculate accuracy, precision, and recall on the grouped
        # dataset
        data, labels = self.read_grouped_data()
        accuracy, precision, recall = self.patientLevelEvaluate(sess, groupPred, data, labels)
        return accuracy, precision, recall
