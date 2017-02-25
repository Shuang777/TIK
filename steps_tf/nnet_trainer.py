import shutil
import os
import time
import numpy as np
import tensorflow as tf
import nnet
import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NNTrainer(object):
  '''
  a class for a neural network that can be used together with Kaldi.
  session is initialized either by read() or by init_nnet().
  '''

  def __init__(self, input_dim, output_dim, batch_size):
    ''' just some basic config for this trainer '''
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.sess = None

  def __exit__ (self):
    if self.sess is not None:
      self.sess.close()


  def read(self, filename):
    ''' read graph from file '''
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.saver = tf.train.import_meta_graph(filename+'.meta')
    
    assert self.sess == None
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      self.saver.restore(self.sess, filename)
      
    self.logits = self.graph.get_collection('logits')[0]
    self.outputs = self.graph.get_collection('outputs')[0]
    self.feats_holder = self.graph.get_collection('feats_holder')[0]
    self.labels_holder = self.graph.get_collection('labels_holder')[0]


  def write(self, filename):
    with self.graph.as_default():
      save_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      saver = tf.train.Saver(save_list, max_to_keep=20)
      saver.save(self.sess, filename)


  def init_nnet(self, nnet_conf, seed=777, init_file = None):
    ''' initializing nnet from file or config (graph creation) '''
    self.graph = tf.Graph()

    with self.graph.as_default():
      feats_holder, labels_holder = nnet.placeholder_inputs(self.input_dim, self.batch_size)

      if init_file is not None:
        logits = nnet.inference_from_file(self.feats_holder, self.input_dim, 
                        self.output_dim, init_file)
      else:
        logits = nnet.inference(feats_holder, self.input_dim, 
                      nnet_conf.get('hidden_units', 1024),
                      nnet_conf.get('num_hidden_layers', 2), 
                      self.output_dim, nnet_conf['nonlin'],
                      nnet_conf.get('batch_norm', False))

      outputs = tf.nn.softmax(logits)
      init_all_op = tf.global_variables_initializer()

      tf.add_to_collection('logits', logits)
      tf.add_to_collection('outputs', outputs)
      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('labels_holder', labels_holder)
    
    assert self.sess == None
    self.sess = tf.Session(graph=self.graph)
    tf.set_random_seed(seed)
    self.sess.run(init_all_op)

    self.logits = logits
    self.outputs = outputs
    self.feats_holder = feats_holder
    self.labels_holder = labels_holder


  def init_training(self, optimizer_conf):
    ''' initialze training graph; 
    assumes self.logits, self.labels_holder in place'''
    with self.graph.as_default():

      loss = nnet.loss(self.logits, self.labels_holder)
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      train_op = nnet.training(optimizer_conf, loss, learning_rate_holder)
      eval_acc = nnet.evaluation(self.logits, self.labels_holder)

    self.loss = loss
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op
    self.eval_acc = eval_acc


  def iter_data(self, logfile, train_gen, learning_rate = None, keep_acc = False):
    '''Train/test one iteration; use learning_rate == None to specify test mode'''

    assert self.batch_size == train_gen.get_batch_size()

    fh = logging.FileHandler(logfile, mode = 'w')
    logger.addHandler(fh)

    sum_avg_loss = 0
    sum_accs = 0
    count_steps = 0

    sum_frames = 0
    sum_acc_frames = 0

    start_time = time.time()
    while(True):

      feed_dict = train_gen.get_batch(self.feats_holder, self.labels_holder)

      if feed_dict is None:   # no more data for training
        break

      feed_dict.update({self.learning_rate_holder: learning_rate})

      if learning_rate is None:
        loss = self.sess.run(self.loss, feed_dict = feed_dict)
      else:
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict = feed_dict)

      sum_avg_loss += loss
      sum_frames += train_gen.get_batch_size()
      duration = time.time() - start_time
      count_steps += 1

      if keep_acc or count_steps % 1000 == 0 or count_steps == 1:
        acc = self.sess.run(self.eval_acc, feed_dict = feed_dict)
        sum_accs += acc
        sum_acc_frames += train_gen.get_batch_size()

        # Print status to stdout.
        logger.info("Step %5d: avg loss = %.6f on %d frames (%.2f sec passed, %.2f frames per sec), peek acc: %.2f%%", 
                    count_steps, sum_avg_loss / count_steps, 
                    sum_frames, duration, sum_frames / duration, 
                    100.0*acc/train_gen.get_batch_size())

    # reset batch_generator because it might be used again
    train_gen.reset_batch()

    avg_loss = sum_avg_loss / count_steps
    if sum_acc_frames == 0:
      avg_acc = None
      avg_acc_str = str(avg_acc)
    else:
      avg_acc = sum_accs/sum_acc_frames
      avg_acc_str = "%.2f%%" % (100.0*avg_acc)

    logger.info("Complete: avg loss = %.6f on %d frames (%.2f sec passed, %.2f frames per sec), peek acc: %s", 
                avg_loss, sum_frames, duration, 
                sum_frames / duration, avg_acc_str)

    logger.removeHandler(fh)

    return avg_loss, avg_acc_str


  def patch_to_batches(self, feats):
    ''' patch data so that it matches our batch_size'''
    if len(feats) % self.batch_size == 0:
      return feats
    row2pad = self.batch_size - (len(feats) % self.batch_size)
    feat_dim = len(feats[0])
    feats_padded = np.vstack([feats, np.zeros([row2pad, feat_dim])])
    return feats_padded
      

  def predict(self, feats, take_log = True):
    posts = []
    for i in range(math.ceil(len(feats) / self.batch_size)):
      batch_start = i*self.batch_size
      batch_end = (i+1)*self.batch_size
      # we avoid copying feats, only patch the last batch
      if len(feats) < batch_end:
        feats_padded = self.patch_to_batches(feats[batch_start:,])
        feed_dict = {self.feats_holder: feats_padded}
      else:
        feed_dict = {self.feats_holder: feats[batch_start:batch_end, :]}

      if take_log:
        batch_posts = self.sess.run(self.outputs, feed_dict=feed_dict)
      else:
        batch_posts = self.sess.run(self.logits, feed_dict=feed_dict)
      posts.append(batch_posts)

    posts = np.vstack(posts)

    return posts[0:len(feats),:]
