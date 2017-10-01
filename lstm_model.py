import os
import time
import pickle
from random import shuffle

import numpy as np
import tensorflow as tf
from sklearn import metrics
from preprocess import padBatch
from createDatasets import splitIntoPatches, splitDataAndLabels

_CHECKPOINT_PREFIX = "./out/params"
_RESULT_PREFIX = "./out/results"


class LstmNet(object):

    """Docstring for LSTM. """

    def __init__(self, mode=1,
                 input_size=13,
                 lstm_state_size=200,
                 num_classes=2,
                 decay=0.01,
                 learning_rate=0.001,
                 output_path=""):
        """TODO: to be defined1.

        :mode: TODO
        :hyper_parms: TODO
        """
        self._mode = mode
        print(input_size)
        # Exposed placeholders
        self.data_ph = tf.placeholder(tf.float32, [None, None, input_size])
        self.target_ph = tf.placeholder(tf.int32, [None])
        self.dropout_ph = tf.placeholder(tf.float32)

        # Exposed ops
        self.inference_op = self._inference(lstm_state_size, num_classes)
        self.loss_op = self._loss(decay)
        self.optimize_op = self._optimize(learning_rate)
        self.raw_prob, self.predict_syllable_op = self._predict_syllable()
        self.predict_patient_op, self.predict_patient_nor_op = self._predict_patient()

    def k_fold_cross_validation(self, dataset,
                                val_dataset=None,
                                folds=10,
                                num_epochs=None,
                                output_path=""):
        """TODO: Docstring for k_fold_cross_validation.

        :data: TODO
        :targets: TODO
        :val_data: TODO
        :val_targets: TODO
        :folds: TODO
        :returns: TODO

        """

        val_set = splitIntoPatches(val_dataset)
        val_set = splitDataAndLabels(val_set)
        val_x, val_y = val_set

        dataset = preprocess.meanAndVarNormalize(dataset, labels=True)

        examples_per_fold = len(dataset) // folds
        fold_datasets = []
        # Create folds
        for i in range(folds):
            index = i*examples_per_fold
            end_index = index + examples_per_fold

            train_set = dataset[:index] + dataset[end_index:]
            train_set = splitIntoPatches(train_set)
            shuffle(train_set)
            train_set = splitDataAndLabels(train_set)
            test_set = dataset[index:end_index]
            syl_test_set = splitIntoPatches(test_set)
            syl_test_set = splitDataAndLabels(syl_test_set)
            pat_test_set = splitDataAndLabels(test_set)
            fold_dataset = (train_set, syl_test_set, pat_test_set)
            fold_datasets.append(fold_dataset)

        sm_preds = np.array([], dtype=np.int32)
        fo_preds = np.array([], dtype=np.int32)
        syl_preds = []
        i = 0
        for train_set, syl_test_set, pat_test_set in fold_datasets:
            # train
            train_x, train_y = train_set
            sess, saver, test_preds = self.train(train_x, train_y,
                                                 val_data=val_x,
                                                 val_targets=val_y,
                                                 test_data=syl_test_set[0],
                                                 num_epochs=num_epochs)
            fold_output_path = "%s/fold%d" % (output_path, i)
            if not os.path.isdir(fold_output_path):
                os.makedirs(fold_output_path)
            test_pred_path = fold_output_path + "/test_preds.pkl"
            param_path = fold_output_path + "/params"
            # Save test predictions and parameters
            with open(test_pred_path, "wb") as f:
                pickle.dump(test_preds, f)
            saver.save(sess, param_path)

            # test
            syl_test_x, syl_test_y = syl_test_set
            pat_test_x, pat_test_y = pat_test_set
            with sess:
                syl_pred = self.syllable_level_evaluate(syl_test_x, syl_test_y,
                                                        probability=True)
                sm_pred, fo_pred = self.patient_level_evaluate(pat_test_x, pat_test_y)
            syl_preds.append(syl_pred)
            sm_preds = np.concatenate((sm_preds, sm_pred))
            fo_preds = np.concatenate((fo_preds, fo_pred))
            i += 1

        return syl_preds, sm_preds, fo_preds


    def train(self, data, target,
              num_epochs=40,
              batch_size=64,
              early_stop_criteria=0.075,
              parameter_path=None,
              val_data=None,
              val_targets=None,
              test_data=None):
        """TODO: Docstring for train.

        :batch_size: TODO
        :session: TODO
        :restore_weights: TODO
        :returns: TODO

        """
        gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        config = tf.ConfigProto(gpu_options=gpu_option)
        session = tf.Session(config=config)
        test_predictions = []
        with session.as_default() as session:
            tf.global_variables_initializer().run()

            params = tf.trainable_variables()
            params = [p for p in params if
                      ("adam" not in p.name and (("weights" in p.name) or "bias" in p.name))]
            saver = tf.train.Saver(params)
            if not parameter_path is None:
                saver.restore(session, parameter_path)

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
                if not test_data is None:
                    test_preds = self.syllable_level_evaluate(test_data, None,
                                                              probability=True)
                    print("it happend")
                    test_predictions.append(test_preds)

                # early stopping
                if not (val_data is None or val_targets is None):
                    val_predictions = self.syllable_level_evaluate(val_data, val_targets)
                    val_prec = metrics.precision_score(val_predictions, val_targets)
                    val_recall = metrics.recall_score(val_predictions, val_targets)
                    f1_score = self._f1_score(val_prec, val_recall)
                    print("Validation f1 score: %s" % f1_score)
                    if f1_score >= max_validation_score:
                        max_validation_score = f1_score
                        momentum_steps = 0
                        saver.save(session, _CHECKPOINT_PREFIX)
                    else:
                        GL = self._generalization_loss(max_validation_score, f1_score)
                        if GL > early_stop_criteria:
                            if momentum_steps > 4:
                                print("Stopping Early. . .")
                                saver.restore(session, _CHECKPOINT_PREFIX)
                                break
                            else:
                                momentum_steps += 1

        return session, saver, test_predictions

    def patient_level_evaluate(self, data,
                               labels,
                               session=None,
                               to_disk=False):
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
        sm_preds = np.zeros([num_examples])
        fo_preds = np.zeros([num_examples])

        if session is None:
            session = tf.get_default_session()
            if session is None:
                raise ValueError('No explicit session argument and no default session')

        with session.as_default() as session:
            print("[*] Starting patient level evaluation")
            for i in range(num_examples):
                batch_x, _ = preprocess.padBatch(data[i])
                feed_dict[self.data_ph] = batch_x
                feed_dict[self.target_ph] = [labels[i]]
                sm_preds[i], fo_preds[i] = session.run([self.predict_patient_op, self.predict_patient_nor_op], feed_dict=feed_dict)

        """
        accuracy = np.sum(predictions == labels) / num_examples
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)

        if to_disk:
            file_suffix = time.strftime('%H:%M:%S', time.gmtime())
            results_save_path = _RESULT_PREFIX + '_patient_' + file_suffix
            with open(results_save_path, 'w') as f:
                f.write("Patient Level Evaluation\nAcc, Prec, Recall\n %f, %f, %f" 
                        % (accuracy, precision, recall))
            print("[*] results saved at %s" % results_save_path)
        """
        print(type(fo_preds))
        return sm_preds, fo_preds

    def syllable_level_evaluate(self, data, labels, session=None, probability=False):
        """
          Evaluates the model per syllable, rather than on a per patient
          level

          session:     Session to run pred_op in
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

        if probability:
            op = self.raw_prob
        else:
            op = self.predict_syllable_op

        with session.as_default() as session:
            print('[*] Starting syllable level evaluation')
            s = 0
            while s < len(data):
                e = min(s + 64, len(data))
                batch_x = data[s : e]
                feed_dict[self.data_ph], _ = preprocess.padBatch(batch_x)
                #feed_dict[self.target_ph] = labels[s : e]
                predictions[s:e] = session.run(op, feed_dict=feed_dict)
                s = e

        """
        accuracy = np.sum(predictions == labels) / len(data)
        precision = metrics.precision_score(labels, predictions)
        recall = metrics.recall_score(labels, predictions)

        if to_disk:
            file_suffix = time.strftime('%H:%M:$S', time.gmtime())
            results_save_path = _RESULT_PREFIX + '_syllable_' + file_suffix
            with open(results_save_path, 'w') as f:
                f.write("Syllable Level Evaluation\nAcc, Prec, Recall\n%f, %f, %f"
                        % (accuracy, precision, recall))
            print("[*] results saved at %s" % results_save_path)
        """

        return predictions


    def _inference(self, lstm_state_size, num_classes):
        """TODO: Docstring for inferenc)

        :mode: TODO
        :returns: TODO

        """
        if self._mode == 1:
            lstm_output = LstmNet._lstm_1(self.data_ph, lstm_state_size)
        elif self._mode == 2:
            lstm_output = LstmNet._lstm_2(self.data_ph, lstm_state_size, self.dropout_ph)
        else:
            lstm_output = LstmNet._bi_lstm_1(self.data_ph, lstm_state_size)

        if self._mode == 3:
            logits = LstmNet._softmax_layer(lstm_output, lstm_state_size*2, num_classes)
        else:
            logits = LstmNet._softmax_layer(lstm_output, lstm_state_size, num_classes)

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
        probs = tf.nn.softmax(self.inference_op)
        positive_probs = tf.squeeze(tf.slice(probs, [0,1], [-1, 1]))
        pred = tf.cast(tf.argmax(self.inference_op, axis=1), "int32")
        return positive_probs, pred


    def _predict_patient(self):
        probs = tf.nn.softmax(self.inference_op)
        true_probs = tf.squeeze(tf.slice(probs, [0, 1], [-1, 1]))
        false_probs = tf.cast(1 - true_probs, tf.float64)
        soft_majority = tf.reduce_mean(true_probs)
        fuzzy_or = 1 - tf.reduce_prod(false_probs)
        return soft_majority, fuzzy_or


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
        lstm_cell = tf.contrib.rnn.LSTMCell(lstm_state_size, initializer=tf.contrib.layers.xavier_initializer())
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
        lstm_cell = tf.contrib.rnn.LSTMCell(lstm_state_size, initializer=tf.contrib.layers.xavier_initializer())
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
        lstm_cell = tf.contrib.rnn.LSTMCell(lstm_state_size, initializer=tf.contrib.layers.xavier_initializer())
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
