import time
from functools import wraps

from sklearn import metrics
import numpy as np
import tensorflow as tf
import preprocess
from createDatasets import splitIntoPatches, splitDataAndLabels

_CHECKPOINT_PATH = "./model_parameters"

class LstmNet(object):

    """Docstring for LSTM. """

    def __init__(self, mode=1,
                 input_size=13,
                 lstm_state_size=200,
                 softmax_hidden_size=200,
                 num_classes=2,
                 decay=0.008,
                 learning_rate=.001):
        """TODO: to be defined1.

        :mode: TODO
        :hyper_parms: TODO
        """
        self._mode = mode

        # Exposed placeholders
        self.data_ph = tf.placeholder(tf.float32, [None, None, input_size])
        self.target_ph = tf.placeholder(tf.int32, [None])
        self.dropout_ph = tf.placeholder(tf.float32)

        # Exposed ops
        self.inference_op = self._inference(lstm_state_size, softmax_hidden_size, num_classes)
        self.loss_op = self._loss(decay)
        self.optimize_op = self._optimize(learning_rate)
        self.predict_syllable_op = self._predict_syllable()
        self.predict_patient_op = self._predict_patient()

    def k_fold_cross_validation(self, dataset, folds=10):
        """TODO: Docstring for k_fold_cross_validation.

        :data: TODO
        :targets: TODO
        :val_data: TODO
        :val_targets: TODO
        :folds: TODO
        :returns: TODO

        """
        examples_per_fold = len(dataset) // folds
        fold_datasets = []
        # Create folds
        for i in range(folds):
            index = i*examples_per_fold
            end_index = index + examples_per_fold
            train_set = dataset[:index] + dataset[:end_index]
            train_set = splitIntoPatches(train_set)
            train_set = splitDataAndLabels(train_set)
            test_set = dataset[index:end_index]
            test_set = splitDataAndLabels(test_set)
            fold_dataset = (train_set, test_set)
            fold_datasets.append(fold_dataset)

        for train_set, test_set in fold_datasets:
            train_x, train_y = train_set
            sess = self.train(train_x, train_y)
            test_x, test_y = test_set
            stats, sess = self.patient_level_evaluate(test_x, test_y, session=sess)
            print('%s, %s, %s' % stats)
            sess.close()


    def train(self, data, target,
              num_epochs=40,
              batch_size=64,
              restore_parameters=False,
              val_data=None,
              val_targets=None,
              test_data=None,
              test_targets=None):
        """TODO: Docstring for train.

        :batch_size: TODO
        :session: TODO
        :restore_weights: TODO
        :returns: TODO

        """
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_option)
        session = tf.Session(config=config)
        with session.as_default() as session:
            tf.global_variables_initializer().run()

            params = tf.trainable_variables()
            params = [p for p in params if
                      ("adam" not in p.name and (("weights" in p.name) or "bias" in p.name))]
            saver = tf.train.Saver(params)
            if restore_parameters:
                saver.restore(session, _CHECKPOINT_PATH)

            feed_dict = dict()
            feed_dict[self.dropout_ph] = 0.5
            max_validation_score = 0
            # Train Loop
            int(num_epochs)
            for i in range(num_epochs):
                print(20 * '*', 'epoch', i+1, 20 * '*')
                start_time = time.time()
                s = 0
                while s < len(data):
                    e = min(s + batch_size, len(data))
                    batch_x = data[s : e]
                    feed_dict[self.data_ph], _ = preprocess.padBatch(batch_x)
                    feed_dict[self.target_ph] = target[s : e]
                    session.run(self.optimize_op, feed_dict=feed_dict)
                    s = e
                end_time = time.time()
                print('the training took: %d(s)' % (end_time - start_time))

                # Show training progress on test set
                if not (test_data is None or test_targets is None):
                    stats, _ = self.syllable_level_evaluate(test_data, test_targets)
                    print('accuracy: %s' % stats[0])

                # early stopping
                if not (val_data is None or val_targets is None):
                    stats, _ = self.syllable_level_evaluate(val_data, val_targets)
                    f1_score = self._f1_score(stats[1], stats[2])
                    if f1_score >= max_validation_score:
                        max_validation_score = f1_score
                        momentum_steps = 0
                        saver.save(session, _CHECKPOINT_PATH)
                    else:
                        GL = self._generalization_loss(max_validation_score, f1_score)
                        print(GL)
                        if GL > .175:
                            if momentum_steps > 3:
                                print("Stopping Early. . .")
                                saver.restore(session, _CHECKPOINT_PATH)
                                break
                            else:
                                momentum_steps += 1

            print("[*] weights saved at %s" % _CHECKPOINT_PATH)
            saver.save(session, _CHECKPOINT_PATH)

        return session

    def patient_level_evaluate(self, data, labels, session=None):
        """Evaluates the model per patient, rather than on a per syllable
        level

        session:     Session to run pred_op in
        pred_op:     Op to get predictions from model
        data:        data to be fed into X placeholder. Should be a list of
                     of lists, where each inner list contains numpy arrays
                     representing the patients' syllables where each row is
                     an MFCC of the syllable
        labels:      labels to be placed into Y

        Returns:     Accuracy, precision, and recall of the model as measured by
                     the models predictions and ground truth labels
        """
        feed_dict = dict()
        feed_dict[self.dropout_ph] = 1
        num_examples = len(data)
        predictions = np.zeros([num_examples])

        if session is None:
            session = tf.get_default_session()
            if session is None:
                raise ValueError('No explicit session argument and no default session')

        with session.as_default() as session:
            print("[*] Starting grouped evaluation")
            for i in range(num_examples):
                batch_x, _ = preprocess.padBatch(data[i])
                feed_dict[self.data_ph] = batch_x
                feed_dict[self.target_ph] = [labels[i]]
                predictions[i] = session.run(self.predict_patient_op, feed_dict=feed_dict)

        accuracy = np.sum(predictions == labels) / num_examples
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)

        return (accuracy, precision, recall), session

    def syllable_level_evaluate(self, data, labels, session=None):
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
        feed_dict[self.dropout_ph] = 1
        predictions = np.zeros([len(data)])

        if session is None:
            session = tf.get_default_session()
            if session is None:
                raise ValueError('No explicit session argument and no default session')

        with session.as_default() as session:
            s = 0
            while s < len(data):
                e = min(s + 64, len(data))
                batch_x = data[s : e]
                feed_dict[self.data_ph], _ = preprocess.padBatch(batch_x)
                feed_dict[self.target_ph] = labels[s : e]
                predictions[s:e] = session.run(self.predict_syllable_op, feed_dict=feed_dict)
                s = e

        accuracy = np.sum(predictions == labels) / len(data)
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)

        return (accuracy, precision, recall), session


    def _inference(self, lstm_state_size, softmax_hidden_size, num_classes):
        """TODO: Docstring for inference.

        :mode: TODO
        :returns: TODO

        """
        if self._mode == 1:
            lstm_output = LstmNet._lstm_1(self.data_ph, lstm_state_size)
        elif self._mode == 2:
            lstm_output = LstmNet._lstm_2(self.data_ph, lstm_state_size, self.dropout_ph)
        else:
            lstm_output = LstmNet._bi_lstm_1(self.data_ph, lstm_state_size)
        logits = LstmNet._softmax_layer(lstm_output, softmax_hidden_size, num_classes)
        return logits


    def _loss(self, decay):
        weights = [param for param in tf.trainable_variables() if "weights" in param.name]
        if self._mode == 3:
            decay /= 2
        regulizer = tf.contrib.layers.l2_regularizer(decay)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.inference_op,
                                                              labels=self.target_ph)
        loss = tf.reduce_mean(loss)

        # Only apply l2 reg to the single-layer LSTM 
        # architectures because we can't do dropout
        if self._mode == 1 or 3:
            loss = loss + tf.contrib.layers.apply_regularization(regulizer, weights)

        return loss


    def _optimize(self, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(self.loss_op)
        return train_op


    def _predict_syllable(self):
        """TODO: Docstring for predict_syllable.
        :returns: TODO

        """
        pred = tf.cast(tf.argmax(self.inference_op, axis=1), "int32")
        return pred


    def _predict_patient(self):
        confidence_levels = tf.nn.softmax(self.inference_op)
        confidence_levels = tf.reduce_sum(confidence_levels, reduction_indices=0)
        return tf.arg_max(confidence_levels, 0)
        """predWeightsRaw = tf.reduce_max(self.inference_op, reduction_indices=1)
        predWeights = tf.nn.softmax(predWeightsRaw)
        weightedPreds = tf.multiply(predWeights, tf.cast(self.predict_syllable_op, 'float'))
        groupPred = tf.round(tf.reduce_sum(weightedPreds))
        return groupPred"""

    @staticmethod
    def _softmax_layer(lstm_output, softmax_hidden_size, num_classes):
        """TODO: Docstring for inference.

        :lstm_output: TODO
        :returns: TODO

        """
        weight_init = tf.random_uniform([softmax_hidden_size, num_classes])
        weights = tf.Variable(weight_init, name="softmax-weights")
        bias = tf.Variable(tf.zeros([num_classes]), name="softmax-bias")
        logits = tf.matmul(lstm_output, weights) + bias
        return logits

    @staticmethod
    def _lstm_1(data, lstm_state_size):
        lengths = LstmNet._actual_lengths(data)
        lstm_cell = tf.contrib.rnn.LSTMCell(lstm_state_size)
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, data, sequence_length=lengths,
                                       dtype=tf.float32)
        # Get the last output for each example
        last_outputs = LstmNet._pluck_last_output(outputs, lengths)
        return last_outputs

    @staticmethod
    def _lstm_2(data, lstm_state_size, dropout_prob):
        """TODO: Docstring for _lstm_2.
        :returns: TODO

        """
        lengths = LstmNet._actual_lengths(data)
        lstm_cell = tf.contrib.rnn.LSTMBlockCell(lstm_state_size)
        dropout_wrapped_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,
                                                             input_keep_prob=dropout_prob,
                                                             output_keep_prob=dropout_prob)
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell, dropout_wrapped_cell])
        outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, data, sequence_length=lengths,
                                       dtype=tf.float32)
        # Get the last output for each example
        last_outputs = LstmNet._pluck_last_output(outputs, lengths)
        return last_outputs

    @staticmethod
    def _bi_lstm_1(data, lstm_state_size):
        """TODO: Docstring for _bi_lstm_1.
        :returns: TODO

        """
        lengths = LstmNet._actual_lengths(data)
        lstm_cell = tf.contrib.rnn.LSTMBlockCell(lstm_state_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell, lstm_cell, data,
                                                     sequence_length=lengths,
                                                     dtype=tf.float32)
        # Get the last output for each example, from the front-pass
        # and the backwards-pass
        last_forward_outputs = LstmNet._pluck_last_output(outputs[0], lengths)
        last_backward_outputs = LstmNet._pluck_last_output(outputs[1], lengths)
        last_outputs = tf.concat([last_forward_outputs, last_backward_outputs], axis=1)
        return last_outputs


    @staticmethod
    def _actual_lengths(sequence):
        """Given a 3D tensor representing a collection of 2D matrices whose
        last few rows may be all zeros, calculate the number of rows that
        actually contain data. The result is a 1D tensor with the number
        of rows containing data for each 2D matrix is sequence.

        :sequence: A 3D tensor whose second dimension may be zero-padded
        """
        used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
        lengths = tf.reduce_sum(used, reduction_indices=1)
        lengths = tf.cast(lengths, tf.int32)
        return lengths

    @staticmethod
    def _pluck_last_output(outputs, example_lengths):
        """Given a 3D tensor representing a collection of each example's
        outputs from each time-step through the LSTM, pluck the last
        relevant output (i.e. not a result of zero-padding) and return
        a 2D tensor representing the last output of each example

        :outputs:           3D tensor of size [batch_size, max_time_step, output_size]
        :example_lengths:   a 1D tensor of size [batch_size] whose elements
                            represent the position of the last relevant time-step's
                            output for each example

        Returns: 2D tensor of size [batch_size, output_size]

        """
        shape = tf.shape(outputs)
        batch_size = shape[0]
        max_length = shape[1]
        out_size = int(outputs.get_shape()[2])

        index = tf.range(0, batch_size) * max_length + (example_lengths - 1)
        flat = tf.reshape(outputs, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    @staticmethod
    def _f1_score(precision, recall):
        """Computes the f1 score given the precision and recall of the model"""
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _generalization_loss(maxMetric, currentMetric):
        """Given the metric from the current training epoch and the best metric
        seen throughout any previous training epoch, computes a measure of the model's
        current generalization performance relative to its best performance seen so far.
        The arguments are usually f1 or accuracy scores computed on the validation set --
        thus serving as estimates to the model's generalization abilities. See "Early
        Stopping -- But When?" (Lutz Prechelt).

        :maxMetric:       The best metric seen during training so far. In the range [0,1]
        :currentMetric:   The metric calculated for the current training epoch. In the range
                          [0, 1].

        """
        minErr = 1 - maxMetric
        currentErr = 1 - currentMetric
        return currentErr / minErr - 1

    @staticmethod
    def parameter_saver():
        """TODO: Docstring for parameter_saver.
        :returns: TODO

        """
        params = tf.trainable_variables()
        filtered_params = []
        for p in params:
            if "adam" not in p.name and (("weights" in p.name) or "bias" in p.name):
                params.append(p)
        saver = tf.train.Saver(filtered_params)
        return saver

    @staticmethod
    def load_pretrained_weights(session=None):
        """TODO: Docstring for load_pretrained_weights.

        :session: TODO
        :returns: TODO

        """
        if session is None:
            session = tf.Session()

        with session.as_default():
            tf.global_variables_initializer().run()
            params = tf.trainable_variables()
            params = [p for p in params if
                      ("adam" not in p.name and (("weights" in p.name) or "bias" in p.name))]
            saver = tf.train.Saver(params)
            saver.restore(session, _CHECKPOINT_PATH)

        return session
