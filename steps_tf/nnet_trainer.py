import shutil
import os
import time
import numpy as np
import tensorflow as tf
import nnet

class Nnet(object):
    '''a class for a neural network that can be used together with Kaldi'''

    def __init__(self, conf, input_dim, output_dim, seed=777):

        #get nnet structure configs
        self.conf = dict(conf.items('nnet'))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_units = 1024
        self.learning_rate = 0.1
        self.batch_size = int(self.conf['batch_size'])

        self.graph = tf.Graph()
        with self.graph.as_default():

            tf.set_random_seed(seed)

            self.feats_holder, self.labels_holder = nnet.placeholder_inputs(self.input_dim, self.batch_size)

            logits = nnet.inference(self.feats_holder, self.input_dim, 
                    self.hidden_units, self.hidden_units, self.output_dim)

            self.loss = nnet.loss(logits, self.labels_holder)

            self.train_op = nnet.training(self.loss, self.learning_rate)

            self.init = tf.initialize_all_variables()

            self.eval_acc = nnet.evaluation(logits, self.labels_holder)

            save_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.saver = tf.train.Saver(save_list)

        self.sess = tf.Session(graph=self.graph)


    def test(self, data_gen):
        sum_avg_loss = 0
        sum_accs = 0
        sum_frames = 0
        start_time = time.time()
        count_steps = 0

        while(data_gen.hasData()):
            feed_dict = data_gen.get_batch(self.feats_holder, self.labels_holder)
            loss, acc = self.sess.run([self.loss, self.eval_acc], feed_dict = feed_dict)
            sum_avg_loss += loss
            sum_accs += acc
            sum_frames += data_gen.get_batch_size()
            count_steps += 1
            if count_steps % 2000 == 0:
                print '%s frames processed' % sum_frames

        duration = time.time() - start_time
        
        print('Test: avg loss = %.6f, frame acc %.2f%%, on %d frames (%.2f sec passed, %.2f frames per sec)' % (sum_avg_loss / count_steps, 100.0 * sum_accs / sum_frames, sum_frames, duration, sum_frames / duration))


    def read(self, filename):
        print 'loading model from %s' % filename
        self.saver.restore(self.sess, filename)


    def write(self, filename):
        self.saver.save(self.sess, filename)
        print 'model saved to %s' % filename


    def init_nnet(self):
        self.sess.run(self.init)


    def train(self, train_gen):
        '''Train one iteration'''
        sum_avg_loss = 0
        sum_accs = 0
        sum_frames = 0
        count_steps = 0
        start_time = time.time()
        while(train_gen.hasData()):

            feed_dict = train_gen.get_batch(self.feats_holder, self.labels_holder)

            _, loss, acc = self.sess.run([self.train_op, self.loss, self.eval_acc], feed_dict = feed_dict)

            sum_avg_loss += loss
            sum_accs += acc
            sum_frames += train_gen.get_batch_size()

            duration = time.time() - start_time
        
            count_steps += 1

            if count_steps % 1000 == 0:
                # Print status to stdout.
                print('Step %d: avg loss = %.6f, frame acc %.2f%%, on %d frames (%.2f sec passed, %.2f frames per sec)' % (count_steps, sum_avg_loss / count_steps, 100.0 * sum_accs / sum_frames, sum_frames, duration, sum_frames / duration))
            
        print('Complete: avg loss = %.6f, frame acc %.2f%%, on %d frames (%.2f sec passed, %.2f frames per sec)' % (sum_avg_loss / count_steps, 100.0 * sum_accs / sum_frames, sum_frames, duration, sum_frames / duration))
