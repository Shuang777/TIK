import shutil
import os
import time
import numpy as np
import tensorflow as tf
from subprocess import Popen,PIPE
import nnet
import math
import logging
from make_nnet_proto import make_nnet_proto, make_lstm_proto

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NNTrainer(object):
  '''
  a class for a neural network that can be used together with Kaldi.
  session is initialized either by read() or by init_nnet().
  '''

  def __init__(self, arch, input_dim, output_dim, batch_size, num_gpus = 1, use_gpu = True,
               summary_dir = None, max_length = None, jitter_window = None):
    ''' just some basic config for this trainer '''
    self.arch = arch
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size
    self.max_length = max_length    # used for lstm
    self.jitter_window = jitter_window # used for jitter training
    self.sess = None
    self.num_gpus = num_gpus
    self.use_gpu = use_gpu
    self.summary_dir = summary_dir


  def make_proto(self, nnet_conf, nnet_proto_file):
    if self.arch == 'dnn':
      make_nnet_proto(self.input_dim, self.output_dim, nnet_conf, nnet_proto_file)
    elif self.arch == 'lstm':
      make_lstm_proto(self.input_dim, self.output_dim, nnet_conf, nnet_proto_file)


  def __exit__ (self):
    if self.sess is not None:
      self.sess.close()

  
  def read(self, filename):
    if self.arch == 'dnn':
      self.read_dnn(filename)
    elif self.arch == 'lstm':
      self.read_lstm(filename)


  def read_dnn(self, filename):
    if self.num_gpus == 1:
      self.read_dnn_single(filename)
    else:
      self.read_dnn_multi(filename)


  def read_dnn_single(self, filename):
    ''' read graph from file '''
    if self.sess is None:
      # this is the first reading
      self.graph = tf.Graph()
      with self.graph.as_default():
        self.saver = tf.train.import_meta_graph(filename+'.meta')

      self.logits = self.graph.get_collection('logits')[0]
      self.outputs = self.graph.get_collection('outputs')[0]
      self.feats_holder = self.graph.get_collection('feats_holder')[0]
      self.labels_holder = self.graph.get_collection('labels_holder')[0]

      self.predict_logits = self.logits     # to keep compatible with multi GPU training code
      self.predict_outputs = self.outputs
      
      self.set_gpu()
      self.sess = tf.Session(graph=self.graph)

    with self.graph.as_default():
      self.saver.restore(self.sess, filename)
      

  def read_dnn_multi(self, filename):
    ''' read graph from file '''
    if self.sess is None:
      # this is the first reading
      self.graph = tf.Graph()
      with self.graph.as_default():
        self.saver = tf.train.import_meta_graph(filename+'.meta')

      self.logits = []
      self.outputs = []
      for i in range(self.num_gpus):
        tower_logits = self.graph.get_collection('logits')[i]
        tower_outputs = self.graph.get_collection('outputs')[i]
        self.logits.append(tower_logits)
        self.outputs.append(tower_outputs)

      self.feats_holder = self.graph.get_collection('feats_holder')[0]
      self.labels_holder = self.graph.get_collection('labels_holder')[0]

      self.predict_logits = self.graph.get_collection('predict_logits')[0]
      self.predict_outputs = self.graph.get_collection('predict_outputs')[0]
      
      self.set_gpu()
      self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
            allow_soft_placement=True) )

    with self.graph.as_default():
      self.saver.restore(self.sess, filename)


  def read_lstm(self, filename):
    if self.num_gpus == 1:
      self.read_lstm_single(filename)
    else:
      self.read_lstm_multi(filename)


  def read_lstm_single(self, filename):
    ''' read graph from file '''
    if self.sess == None:
      self.graph = tf.Graph()
      with self.graph.as_default():
        self.saver = tf.train.import_meta_graph(filename+'.meta')
     
      self.logits = self.graph.get_collection('logits')[0]
      self.outputs = self.graph.get_collection('outputs')[0]
      self.feats_holder = self.graph.get_collection('feats_holder')[0]
      self.labels_holder = self.graph.get_collection('labels_holder')[0]
      self.seq_length_holder = self.graph.get_collection('seq_length_holder')[0]
      self.mask_holder = self.graph.get_collection('mask_holder')[0]
      self.keep_in_prob_holder = self.graph.get_collection('keep_in_prob_holder')[0]
      self.keep_out_prob_holder = self.graph.get_collection('keep_out_prob_holder')[0]

      self.predict_logits = self.logits     # to keep compatible with multi GPU training code
      self.predict_outputs = self.outputs

      self.set_gpu()
      self.sess = tf.Session(graph=self.graph)
      
    with self.graph.as_default():
      self.saver.restore(self.sess, filename)
    
  
  def read_lstm_multi(self, filename):
    if self.sess == None:
      self.graph = tf.Graph()
      with self.graph.as_default():
        self.saver = tf.train.import_meta_graph(filename+'.meta')
     
      self.logits = []
      self.outputs = []
      for i in range(self.num_gpus):
        tower_logits = self.graph.get_collection('logits')[i]
        tower_outputs = self.graph.get_collection('outputs')[i]
        self.logits.append(tower_logits)
        self.outputs.append(tower_outputs)

      self.feats_holder = self.graph.get_collection('feats_holder')[0]
      self.labels_holder = self.graph.get_collection('labels_holder')[0]
      self.seq_length_holder = self.graph.get_collection('seq_length_holder')[0]
      self.mask_holder = self.graph.get_collection('mask_holder')[0]
      self.keep_in_prob_holder = self.graph.get_collection('keep_in_prob_holder')[0]
      self.keep_out_prob_holder = self.graph.get_collection('keep_out_prob_holder')[0]

      self.predict_logits = self.graph.get_collection('predict_logits')[0]
      self.predict_outputs = self.graph.get_collection('predict_outputs')[0]

      self.set_gpu()
      self.sess = tf.Session(graph=self.graph)
    
    with self.graph.as_default():
      self.saver.restore(self.sess, filename)
  

  def write(self, filename):
    with self.graph.as_default():
      save_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      saver = tf.train.Saver(save_list, max_to_keep=20)
      saver.save(self.sess, filename)


  def set_gpu(self):
    if self.use_gpu and self.num_gpus != 0:
      p1 = Popen (['pick-gpu', str(self.num_gpus)], stdout=PIPE)
      gpu_ids = str(p1.stdout.read(), 'utf-8')
      if gpu_ids == "-1":
        raise RuntimeError("Picking gpu failed")
      os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    else:
      os.environ['CUDA_VISIBLE_DEVICES'] = ''


  def get_num_subnnets(self):
    return len(self.feats_holder)


  def init_nnet(self, nnet_proto_file):
    if self.arch == 'dnn':
      self.init_dnn(nnet_proto_file)
    elif self.arch == 'lstm':
      self.init_lstm(nnet_proto_file)


  def init_dnn(self, nnet_proto_file, seed = 777):
    if self.num_gpus == 1:
      self.init_dnn_single(nnet_proto_file, seed)
    else:
      self.init_dnn_multi(nnet_proto_file, seed)


  def init_dnn_single(self, nnet_proto_file, seed=777):
    ''' initializing nnet from file or config (graph creation) '''
    self.graph = tf.Graph()

    with self.graph.as_default():
      tf.set_random_seed(seed)
      feats_holder, labels_holder = nnet.placeholder_dnn(self.input_dim, self.batch_size)
      
      logits = nnet.inference_dnn(feats_holder, nnet_proto_file)
      outputs = tf.nn.softmax(logits)
      init_all_op = tf.global_variables_initializer()

      tf.add_to_collection('logits', logits)
      tf.add_to_collection('outputs', outputs)
      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('labels_holder', labels_holder)
    
    assert self.sess == None
    self.set_gpu()
    self.sess = tf.Session(graph=self.graph)
    self.sess.run(init_all_op)

    self.logits = logits
    self.outputs = outputs
    self.feats_holder = feats_holder
    self.labels_holder = labels_holder

    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.graph)
      self.summary_writer.flush()

    return

  def init_dnn_multi(self, nnet_proto_file, seed=777):
    ''' initializing nnet from file or config (graph creation) '''
    self.graph = tf.Graph()
    self.set_gpu()
    self.logits = []
    self.outputs = []

    with self.graph.as_default(), tf.device('/cpu:0'):
      tf.set_random_seed(seed)

      feats_holder, labels_holder = nnet.placeholder_dnn(
                                      self.input_dim, 
                                      self.batch_size*self.num_gpus)
      
      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('labels_holder', labels_holder)
      self.feats_holder = feats_holder
      self.labels_holder = labels_holder

      for i in range(self.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % i) as scope:
            
            tower_start_index = i * self.batch_size
            tower_end_index = (i+1) * self.batch_size

            tower_feats_holder = feats_holder[tower_start_index:tower_end_index,:]
            tower_logits = nnet.inference_dnn(tower_feats_holder, nnet_proto_file, reuse = (i != 0))
            tower_outputs = tf.nn.softmax(tower_logits)

            self.logits.append(tower_logits)
            self.outputs.append(tower_outputs)

            tf.add_to_collection('logits', tower_logits)
            tf.add_to_collection('outputs', tower_outputs)

      # end towers/gpus
      init_all_op = tf.global_variables_initializer()
      predict_logits = nnet.inference_dnn(feats_holder, nnet_proto_file, reuse = True)
      predict_outputs = tf.nn.softmax(predict_logits)
      tf.add_to_collection('predict_logits', predict_logits)
      tf.add_to_collection('predict_outputs', predict_outputs)

    # end tf graph

    assert self.sess == None
    self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
                           allow_soft_placement=True))
    self.sess.run(init_all_op)

    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.graph)
      self.summary_writer.flush()

    return
 
  def init_lstm(self, nnet_proto_file, seed = 777):
    if self.num_gpus == 1:
      self.init_lstm_single(nnet_proto_file, seed)
    else:
      self.init_lstm_multi(nnet_proto_file, seed)


  def init_lstm_single(self, nnet_proto_file, seed):
    self.graph = tf.Graph()
    with self.graph.as_default():
      feats_holder, seq_length_holder, mask_holder, labels_holder = nnet.placeholder_lstm(self.input_dim, 
                                                                             self.max_length,
                                                                             self.batch_size)
      
      keep_in_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_in_prob')
      keep_out_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_out_prob')

      logits = nnet.inference_lstm(feats_holder, seq_length_holder, nnet_proto_file, 
                                   keep_in_prob_holder, keep_out_prob_holder)

      outputs = tf.nn.softmax(logits)
      init_all_op = tf.global_variables_initializer()

      tf.add_to_collection('logits', logits)
      tf.add_to_collection('outputs', outputs)
      tf.add_to_collection('feats_holder', feats_holder)
      tf.add_to_collection('labels_holder', labels_holder)
      tf.add_to_collection('seq_length_holder', seq_length_holder)
      tf.add_to_collection('mask_holder', mask_holder)
      tf.add_to_collection('keep_in_prob_holder', keep_in_prob_holder)
      tf.add_to_collection('keep_out_prob_holder', keep_out_prob_holder)

    
    assert self.sess == None
    self.set_gpu()
    self.sess = tf.Session(graph=self.graph)
    tf.set_random_seed(seed)
    self.sess.run(init_all_op)

    self.logits = logits
    self.outputs = outputs
    self.feats_holder = feats_holder
    self.labels_holder = labels_holder
    self.seq_length_holder = seq_length_holder
    self.mask_holder = mask_holder
    self.keep_in_prob_holder = keep_in_prob_holder
    self.keep_out_prob_holder = keep_out_prob_holder

    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.graph)
      self.summary_writer.flush()


  def init_lstm_multi(self, nnet_proto_file, seed):
    self.graph = tf.Graph()
    self.set_gpu()
    self.logits = []
    self.outputs = []
   
    with self.graph.as_default(), tf.device('/cpu:0'):

      feats_holder, seq_length_holder, \
        mask_holder, labels_holder = nnet.placeholder_lstm(
                                         self.input_dim,
                                         self.max_length,
                                         self.batch_size*self.num_gpus)
      
      keep_in_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_in_prob')
      keep_out_prob_holder = tf.placeholder(tf.float32, shape=[], name = 'keep_out_prob')
      tf.add_to_collection('keep_in_prob_holder', keep_in_prob_holder)
      tf.add_to_collection('keep_out_prob_holder', keep_out_prob_holder)
      self.keep_in_prob_holder = keep_in_prob_holder
      self.keep_out_prob_holder = keep_out_prob_holder

      for i in range(self.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % i) as scope:
            
            tower_start_index = i * self.batch_size
            tower_end_index = (i+1) * self.batch_size

            tower_feats_holder = feats_holder[tower_start_index:tower_end_index,:,:]
            tower_seq_length_holder = seq_length_holder[tower_start_index:tower_end_index]

            tower_logits = nnet.inference_lstm(tower_feats_holder, tower_seq_length_holder, nnet_proto_file, 
                                         keep_in_prob_holder, keep_out_prob_holder, reuse = (i != 0))

            tower_outputs = tf.nn.softmax(tower_logits)

            tf.add_to_collection('logits', tower_logits)
            tf.add_to_collection('outputs', tower_outputs)
            self.logits.append(tower_logits)
            self.outputs.append(tower_outputs)

      # end towers/gpus

      init_all_op = tf.global_variables_initializer()

    # end tf graphs

    tf.add_to_collection('feats_holder', feats_holder)
    tf.add_to_collection('labels_holder', labels_holder)
    tf.add_to_collection('seq_length_holder', seq_length_holder)
    tf.add_to_collection('mask_holder', mask_holder)
    self.feats_holder = feats_holder
    self.labels_holder = labels_holder
    self.seq_length_holder = seq_length_holder
    self.mask_holder = mask_holder

    assert self.sess == None
    self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
                           allow_soft_placement=True))
    tf.set_random_seed(seed)
    self.sess.run(init_all_op)

    if self.summary_dir is not None:
      self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.graph)
      self.summary_writer.flush()


  def init_training(self, optimizer_conf):
    if self.arch == 'dnn':
      self.init_training_dnn(optimizer_conf)
    elif self.arch == 'lstm':
      self.init_training_lstm(optimizer_conf)


  def init_training_dnn(self, optimizer_conf, seed = 777):
    if self.num_gpus == 1:
      self.init_training_dnn_single(optimizer_conf)
    else:
      self.init_training_dnn_multi(optimizer_conf)


  def init_training_dnn_single(self, optimizer_conf):
    ''' initialze training graph; 
    assumes self.logits, self.labels_holder in place'''
    with self.graph.as_default():

      loss = nnet.loss_dnn(self.logits, self.labels_holder)
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      train_op = nnet.training(optimizer_conf, loss, learning_rate_holder)
      eval_acc = nnet.evaluation_dnn(self.logits, self.labels_holder)
      
    self.loss = loss
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op
    self.eval_acc = eval_acc


  def init_training_dnn_multi(self, optimizer_conf):
    tower_losses = []
    tower_grads = []
    tower_accs = []

    with self.graph.as_default(), tf.device('/cpu:0'):
      
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      assert optimizer_conf['op_type'].lower() == 'sgd'
      opt = tf.train.GradientDescentOptimizer(learning_rate_holder)

      for i in range(self.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % (i)) as scope:
            
            tower_start_index = i*self.batch_size
            tower_end_index = (i+1)*self.batch_size

            tower_labels_holder = self.labels_holder[tower_start_index:tower_end_index]

            loss = nnet.loss_dnn(self.logits[i], tower_labels_holder)
            tower_losses.append(loss)
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)
            eval_acc = nnet.evaluation_dnn(self.logits[i], tower_labels_holder)
            tower_accs.append(eval_acc)

      grads = nnet.average_gradients(tower_grads)
      train_op = opt.apply_gradients(grads)
      losses = tf.reduce_sum(tower_losses)
      accs = tf.reduce_sum(tower_accs)

    self.loss = losses
    self.eval_acc = accs
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op


  def init_training_lstm(self, optimizer_conf):
    if self.num_gpus == 1:
      self.init_training_lstm_single(optimizer_conf)
    else:
      self.init_training_lstm_multi(optimizer_conf)


  def init_training_lstm_single(self, optimizer_conf):
    ''' initialze training graph; 
    assumes self.logits, self.labels_holder in place'''
    with self.graph.as_default():

      loss = nnet.loss_lstm(self.logits, self.labels_holder, self.mask_holder)
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      train_op = nnet.training(optimizer_conf, loss, learning_rate_holder)
      eval_acc = nnet.evaluation_lstm(self.logits, self.labels_holder, self.mask_holder)

    self.loss = loss
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op
    self.eval_acc = eval_acc

 
  def init_training_lstm_multi(self, optimizer_conf):
    ''' initialze training graph; 
    assumes self.logits, self.labels_holder in place'''
    
    tower_losses = []
    tower_grads = []
    tower_accs = []

    with self.graph.as_default(), tf.device('/cpu:0'):
      
      learning_rate_holder = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
      assert optimizer_conf['op_type'].lower() == 'sgd'
      opt = tf.train.GradientDescentOptimizer(learning_rate_holder)

      for i in range(self.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('Tower_%d' % (i)) as scope:
            
            tower_start_index = i*self.batch_size
            tower_end_index = (i+1)*self.batch_size

            tower_mask_holder = self.mask_holder[tower_start_index:tower_end_index,:]
            tower_labels_holder = self.labels_holder[tower_start_index:tower_end_index,:]

            loss = nnet.loss_lstm(self.logits[i], tower_labels_holder, tower_mask_holder)
            tower_losses.append(loss)
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)
            eval_acc = nnet.evaluation_lstm(self.logits[i], tower_labels_holder, tower_mask_holder)
            tower_accs.append(eval_acc)

      grads = nnet.average_gradients(tower_grads)
      train_op = opt.apply_gradients(grads)
      losses = tf.reduce_sum(tower_losses)
      accs = tf.reduce_sum(tower_accs)

    self.loss = losses
    self.eval_acc = accs
    self.learning_rate_holder = learning_rate_holder
    self.train_op = train_op

 
  def prepare_feed(self, train_gen, learning_rate, keep_in_prob, keep_out_prob):
    if self.arch == 'dnn':
      feed_dict, has_data = self.prepare_feed_dnn(train_gen, learning_rate)
    elif self.arch == 'lstm':
      feed_dict, has_data = self.prepare_feed_lstm(train_gen, learning_rate, keep_in_prob, keep_out_prob)
      feed_dict.update( {
                  self.keep_in_prob_holder: keep_in_prob,
                  self.keep_out_prob_holder: keep_out_prob } )

    feed_dict.update( { self.learning_rate_holder: learning_rate })

    return feed_dict, has_data


  def prepare_feed_dnn(self, train_gen, learning_rate):

    x, y = train_gen.get_batch_frames()

    feed_dict = { self.feats_holder : x,
                  self.labels_holder : y }
    return feed_dict, x is not None


  def prepare_feed_lstm(self, train_gen, learning_rate, keep_in_prob, keep_out_prob):
    x, y, seq_length, mask = train_gen.get_batch_utterances()

    feed_dict = { self.feats_holder : x,
                  self.labels_holder : y,
                  self.seq_length_holder: seq_length,
                  self.mask_holder: mask }
    return feed_dict, x is not None
    

  def iter_data(self, logfile, train_gen, learning_rate = None, keep_acc = False, 
                keep_in_prob = 1.0, keep_out_prob = 1.0):
    '''Train/test one iteration; use learning_rate == None to specify test mode'''

    assert self.batch_size*self.num_gpus == train_gen.get_batch_size()

    fh = logging.FileHandler(logfile, mode = 'w')
    logger.addHandler(fh)

    sum_avg_loss = 0
    sum_accs = 0
    count_steps = 0

    sum_frames = 0
    sum_acc_frames = 0

    start_time = time.time()

    while(True):

      feed_dict, has_data = self.prepare_feed(train_gen, learning_rate, keep_in_prob, keep_out_prob)

      if not has_data:   # no more data for training
        break

      if learning_rate is None:
        loss = self.sess.run(self.loss, feed_dict = feed_dict)
      else:
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict = feed_dict)

      batch_frames = train_gen.get_last_batch_frames()
      sum_avg_loss += loss
      sum_frames += batch_frames
      duration = time.time() - start_time
      count_steps += 1

      if keep_acc or count_steps % 1000 == 0 or count_steps == 1:
        acc = self.sess.run(self.eval_acc, feed_dict = feed_dict)
        sum_accs += acc
        sum_acc_frames += train_gen.get_last_batch_frames()

        # Print status to stdout.
        if count_steps % 1000 == 0:
          logger.info("Step %5d: avg loss = %.6f on %d frames (%.2f sec passed, %.2f frames per sec), peek acc: %.2f%%", 
                    count_steps, sum_avg_loss / (count_steps*self.num_gpus), 
                    sum_frames, duration, sum_frames / duration, 
                    100.0*acc/train_gen.get_last_batch_frames())

    # reset batch_generator because it might be used again
    train_gen.reset_batch()

    avg_loss = sum_avg_loss / (count_steps * self.num_gpus)
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


  def predict(self, feats, take_log = True):
    if self.arch == 'dnn':
      posts = self.predict_dnn(feats, take_log)
    elif self.arch == 'lstm':
      posts = self.predict_lstm(feats, take_log)

    return posts


  def patch_to_batches(self, feats):
    ''' patch data so that it matches our batch_size'''
    if len(feats) % self.batch_size == 0:
      return feats
    row2pad = self.batch_size - (len(feats) % self.batch_size)
    feat_dim = len(feats[0])
    feats_padded = np.vstack([feats, np.zeros([row2pad, feat_dim])])
    return feats_padded
      

  def predict_dnn(self, feats, take_log = True):
    '''
    args: 
      feats: np 2-d array of size[num_frames, feat_dim]
    output:
      posts: np 2-d array of size[num_frames, num_targets]
    '''
    posts = []
    for i in range(math.ceil(len(feats) / self.batch_size)):
      batch_start = i*self.batch_size
      batch_end = (i+1)*self.batch_size
      # we avoid copying feats, only patch the last batch
      if len(feats) < batch_end:
        feats_padded = self.patch_to_batches(feats[batch_start:,])
      else:
        feats_padded = feats[batch_start:batch_end, :]
      
      feed_dict = {self.feats_holder: feats_padded}

      if take_log:
        batch_posts = self.sess.run(self.predict_outputs, feed_dict=feed_dict)
      else:
        batch_posts = self.sess.run(self.predict_logits, feed_dict=feed_dict)
      posts.append(batch_posts)

    posts = np.vstack(posts)

    return posts[0:len(feats),:]

  
  def pack_utterance(self, feats):
    '''
    args:
      feats: list of array, i.e. matrix of size [num_frames, feat_dim]
    output:
      feat_packs: np 3-d array of size [num_batches, max_length, feat_dim]
      seq_length: np array of size [num_batches]
    '''
    max_length = self.max_length
    jitter_window = self.jitter_window
    start_index = 0
    feats_packed = []
    seq_length = []
    post_pick = []
    pick_start = 0
    pick_end = (max_length + jitter_window) // 2
    while start_index + max_length < len(feats):
      end_index = start_index + max_length
      feats_packed.append(feats[start_index:end_index])
      seq_length.append(max_length)
      post_pick.append([pick_start, pick_end])
      # only the first window starts from 0, all others start from (max_length - jittter_window) / 2
      pick_start = (max_length - jitter_window) // 2      
      start_index += jitter_window

    num_zero = max_length + start_index - len(feats)
    zeros2pad = np.zeros((num_zero, len(feats[0])))
    feats_packed.append(np.concatenate((feats[start_index:], zeros2pad)))
    seq_length.append(len(feats) - start_index)
    # our last window goes till the end of the utterance
    post_pick.append([pick_start, len(feats) - start_index])

    # now we need to pad more zeros to fit the place holder, because each place holder can only host [ batch_size x max_length x feat_dim ] this many data
    baches2pad = self.batch_size - len(feats_packed) % self.batch_size
    if baches2pad != 0:
      zeros2pad = np.zeros((max_length, len(feats[0])))
      for i in range(baches2pad):
        feats_packed.append(zeros2pad)
        seq_length.append(0)
        post_pick.append([0, 0])

    feats_packed = np.array(feats_packed)
    seq_length = np.array(seq_length)

    return feats_packed, seq_length, post_pick


  def predict_lstm(self, feats, take_log = True):
    '''
    we need a sliding window to generate frame posteriors

    args: 
      feats: np 2-d array of size[num_frames, feat_dim]
    output:
      posts: np 2-d array of size[num_frames, num_targets]
    '''
    # we use a rolling window to process the whole utterance
    feats_packed, seq_length, post_pick = self.pack_utterance(feats)

    posts = []
    assert len(feats_packed) % self.batch_size == 0
    for i in range(len(feats_packed) // self.batch_size):
      batch_start = i*self.batch_size
      batch_end = (i+1)*self.batch_size
      feats_batch = feats_packed[batch_start:batch_end, :]
      seq_length_batch = seq_length[batch_start:batch_end]

      feed_dict = {self.feats_holder: feats_batch,
                   self.seq_length_holder: seq_length_batch,
                   self.keep_in_prob_holder: 1.0,
                   self.keep_out_prob_holder: 1.0}

      if take_log:
        batch_posts = self.sess.run(self.predict_outputs, feed_dict=feed_dict)
      else:
        batch_posts = self.sess.run(self.predict_logits, feed_dict=feed_dict)
      # batch_posts of size [batch_size, max_len, num_targets]

      for piece in range(self.batch_size):
        if post_pick[piece][0] != post_pick[piece][1]:
          # post_pick specifies the index of posterior to pick out to form decoding sequence
          posts.append(batch_posts[piece, post_pick[piece][0]:post_pick[piece][1]])

    posts = np.concatenate(posts)

    return posts[0:len(feats),:]
